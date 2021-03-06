import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm, trange

import compression.logistic
import compression.models
from compression.bitstream import Bitstream, CompressionModel
from compression.blackbox import BlackBoxBitstream
from compression.utils import setup, load_image_and_crop, make_testing_dataloader


def main_compression_test(*, stream: Bitstream, model: CompressionModel, dataloader, device):
    """
    Runs the encode-then-decode pipeline and checks that the data has been recovered
    """

    init_stream_len = len(stream)
    num_batches = len(dataloader)
    encoded_dbg_info = []
    encoded_x_raw = []
    shortest_post_decode_len = np.inf  # shortest stream length after any decode, for monitoring required initial bits
    num_symbols_so_far = 0
    total_expected_bits = 0.
    batch_expected_bpds = []
    batch_actual_bpds = []

    # Encode
    for i_batch, (x_raw,) in enumerate(tqdm(dataloader, desc='encoding')):
        curr_bits_before = len(stream)
        encoded_x_raw.append(x_raw)
        x_raw = x_raw.to(device=device)
        dbg_info = model.encode(x_raw, stream=stream)
        encoded_dbg_info.append(dbg_info)
        shortest_post_decode_len = min(shortest_post_decode_len, min(dbg_info['post_decode_lengths']))
        curr_bits_after = len(stream)
        curr_num_symbols = int(np.prod(x_raw.shape))

        # Record expected bpd from the model logp calculation
        expected_bits = model.forward(x_raw.to(dtype=torch.float64))['total_logd'].sum().item() / (-math.log(2.))
        batch_expected_bpds.append(expected_bits / curr_num_symbols)
        batch_actual_bpds.append((curr_bits_after - curr_bits_before) / curr_num_symbols)
        total_expected_bits += expected_bits
        num_symbols_so_far += curr_num_symbols
        if i_batch % 10 == 0:
            print(
                '>>> {num_so_far:03d}/{num_batches} expected {expected:.05f} actual {actual:.05f} avg {avg:.05f} init_bits_needed {init_bits_needed:d}'.format(
                    num_so_far=i_batch + 1,
                    num_batches=num_batches,
                    expected=batch_expected_bpds[-1],
                    actual=batch_actual_bpds[-1],
                    avg=(curr_bits_after - init_stream_len) / num_symbols_so_far,
                    init_bits_needed=init_stream_len - shortest_post_decode_len
                ))
    assert len(batch_expected_bpds) == len(batch_actual_bpds)
    results = {
        'expected_batch_bpds': list(map(float, batch_expected_bpds)),
        'actual_batch_bpds': list(map(float, batch_actual_bpds)),
        'init_bits_needed': init_stream_len - shortest_post_decode_len,
        'single_expected_bpd': total_expected_bits / num_symbols_so_far,  # this mean taken over individual datapoints
        'single_actual_bpd': (len(stream) - init_stream_len) / num_symbols_so_far,  # same here
    }
    print(
        f'===== total bpd expected {results["single_expected_bpd"]:0.5f} '
        f'actual {results["single_actual_bpd"]:.05f} ====='
    )

    # Verify decoding
    decoding_passed = True
    for i_batch in trange(num_batches, desc='decoding'):
        dbg_info = encoded_dbg_info[-(i_batch + 1)]
        x_raw = model.decode(bs=dbg_info['z_sym'].shape[0], stream=stream, encoding_dbg_info=dbg_info)
        if not (x_raw == encoded_x_raw[-(i_batch + 1)].to(device=device)).all():
            decoding_passed = False
            break
    if len(stream) != init_stream_len:
        decoding_passed = False
    print('recovered! {}'.format(num_batches) if decoding_passed else 'decoding failed!')
    results['decoding_passed'] = decoding_passed

    return results


def main_timing_test(batches, model, stream: Bitstream, device):
    """
    Timing test for the compositional algorithm
    """

    batches = [b.to(device=device) for b in batches]
    datapoints = torch.cat(batches, 0)

    # Running the model by itself
    fwd_times = []
    fwd_results = []
    for x_raw in tqdm(batches, desc='fwd'):
        tstart = time.time()
        fwd_results.append(model.forward(x_raw.to(dtype=torch.float64)))
        fwd_times.append(time.time() - tstart)
    inv_times = []
    for res in tqdm(fwd_results, desc='inv'):
        tstart = time.time()
        x, _ = model.main_flow(res['z'], inverse=True, aux=None)
        model.dequant_flow(res['u'], inverse=True, aux=x.floor())  # TODO check closeness to epsilon?
        inv_times.append(time.time() - tstart)

    print('fwd times', fwd_times)
    print('inv times', inv_times)
    print('fwd: {} +/- {} sec'.format(np.mean(fwd_times[1:]), np.std(fwd_times[1:])))
    print('inv: {} +/- {} sec'.format(np.mean(inv_times[1:]), np.std(inv_times[1:])))

    # Coding with the compositional algorithm
    enc_times = []
    encoded_dbg_info = []
    for x_raw in tqdm(batches, desc='encoding'):
        tstart = time.time()
        encoded_dbg_info.append(model.encode(x_raw, stream=stream))
        enc_times.append(time.time() - tstart)
    dec_times = []
    decoded_batches = []
    for dbg_info in tqdm(encoded_dbg_info[::-1], desc='decoding'):
        tstart = time.time()
        decoded_batches.append(model.decode(bs=dbg_info['z_sym'].shape[0], stream=stream, encoding_dbg_info=None))
        dec_times.append(time.time() - tstart)
    assert torch.allclose(datapoints, torch.cat(decoded_batches[::-1], 0))

    print('enc times', enc_times)
    print('dec times', dec_times)
    print('enc: {} +/- {} sec'.format(np.mean(enc_times[1:]), np.std(enc_times[1:])))
    print('dec: {} +/- {} sec'.format(np.mean(dec_times[1:]), np.std(dec_times[1:])))


def main_timing_test_custom(batches, model, stream: Bitstream, device):
    """
    Timing test for the compositional algorithm
    """

    batches = [b.to(device=device) for b in batches]
    datapoints = torch.cat(batches, 0)

    # Coding with the compositional algorithm
    enc_times = []
    encoded_dbg_info = []
    for x_raw in tqdm(batches, desc='encoding'):
        tstart = time.time()
        encoded_dbg_info.append(model.encode(x_raw, stream=stream))
        enc_times.append(time.time() - tstart)
    dec_times = []
    decoded_batches = []
    for dbg_info in tqdm(encoded_dbg_info[::-1], desc='decoding'):
        tstart = time.time()
        decoded_batches.append(model.decode(bs=dbg_info['z_sym'].shape[0], stream=stream, encoding_dbg_info=None))
        dec_times.append(time.time() - tstart)
    if not torch.allclose(datapoints, torch.cat(decoded_batches[::-1], 0)):
        print('Image failed')

    print('enc times', enc_times)
    print('dec times', dec_times)
    print('enc: {} +/- {} sec'.format(np.mean(enc_times[1:]), np.std(enc_times[1:])))
    print('dec: {} +/- {} sec'.format(np.mean(dec_times[1:]), np.std(dec_times[1:])))



def main_timing_test_blackbox(datapoints, model: CompressionModel, bbstream: BlackBoxBitstream):
    """
    Timing test for the black box algorithm
    """

    enc_times = []
    for single_x_raw in tqdm(datapoints, desc='encoding (black box)'):
        init_bits = len(bbstream)
        tstart = time.time()
        bbstream.encode(single_x_raw)
        enc_times.append(time.time() - tstart)
        print('expected {expected:.05f} actual {actual:.05f} enc_time {enc_time:.05f}'.format(
            expected=model.forward(single_x_raw[None].to(dtype=torch.float64))['total_logd'][0].item() / (
                    -math.log(2.) * int(np.prod(single_x_raw.shape))),
            actual=(len(bbstream) - init_bits) / int(np.prod(single_x_raw.shape)),
            enc_time=enc_times[-1]
        ))

    dec_times = []
    decoded_datapoints = []
    for _ in tqdm(reversed(datapoints), desc='decoding (black box)'):
        tstart = time.time()
        decoded_datapoints.append(bbstream.decode().cpu())
        dec_times.append(time.time() - tstart)

    print('enc: {} +/- {} sec'.format(np.mean(enc_times[1:]), np.std(enc_times[1:])))
    print('dec: {} +/- {} sec'.format(np.mean(dec_times[1:]), np.std(dec_times[1:])))

    assert torch.allclose(datapoints.cpu(), torch.stack(decoded_datapoints[::-1], 0))


def main_val(dataloader, model, device):
    """
    Negative log likelihood evaluation only; no compression
    """

    start_time = time.time()
    all_bpds = []
    for batch, in tqdm(dataloader):
        batch_result = model.forward(batch.to(dtype=torch.float64, device=device))
        all_bpds.extend((batch_result['total_logd'] / (-math.log(2.) * int(np.prod(batch.shape[1:])))).tolist())

    bpd = np.mean(all_bpds)
    std = np.std(all_bpds)
    tmp = time.time() - start_time
    num_crops = len(all_bpds)

    print(f'overall bpd: {bpd} +/- {std}, total time: {tmp}, num datapoints: {num_crops}')
    return [bpd, std, tmp, len(all_bpds)]


def main():
    parser = argparse.ArgumentParser()
    # Common arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encode_bs', type=int, default=64)
    parser.add_argument('--black_box_jacobian_bs', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', default='val_only', type=str)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('--single_image', action='store_true')
    parser.add_argument('--limit_dataset_size', type=int, default=None)
    parser.add_argument('--imagenet64_data_path', type=str, default='~/data/imagenet-small/valid_64x64.npy')
    parser.add_argument('--imagenet64_model', type=str, default='~/data/flowpp_imagenet64_model.npz')
    # Script mode
    parser.add_argument('--test_output_filename', type=str, default=None)
    parser.add_argument('--timing_test_count', type=int, default=6)
    # Default compression options
    parser.add_argument('--neg_log_noise_scale', type=int, default=14)
    parser.add_argument('--disc_bits', type=int, default=32)
    parser.add_argument('--disc_range', type=int, default=256)
    parser.add_argument('--ans_init_bits', type=int, default=10000000)
    parser.add_argument('--ans_num_streams', type=int, default=16)
    parser.add_argument('--ans_mass_bits', type=int, default=60)  # probably never need to change this
    args = parser.parse_args()

    setup(seed=args.seed)

    # Load data
    model_ctor = compression.models.load_imagenet64_model
    model_filename = os.path.expanduser(args.imagenet64_model)
    args.input = os.path.expanduser(args.input)

    if args.single_image:
        image_list = [args.input]
    else:
        image_list = glob.glob(args.input + '/*.png')

    # Load model
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = model_ctor(model_filename, force_float32_cond=True).to(device=device)

    assert len(image_list) > 0

    if args.mode == 'val_only':
        results = []

        for i, filename in enumerate(image_list):
            print(f'Processing image {i + 1}/{len(image_list)} - {filename}')
            dataset, bs = load_image_and_crop(filename)

            dataloader, dataset = make_testing_dataloader(
                dataset, seed=args.seed, limit_dataset_size=args.limit_dataset_size, bs=64
            )

            # Dispatch to the chosen mode's main function
            result = main_val(dataloader, model, device)
            assert result[3] == len(dataset)

            results.append(result)

        results = np.array(results)
        means = np.sum(results, axis=0) / len(image_list)

        print('OVERALL RESULTS:')
        print(f'bpd: {means[0]} +/- {means[1]}, total time: {means[2]}, num datapoints: {means[3]}')
    else:
        def _make_stream(total_init_bits_=None):
            return Bitstream(
                device=device,
                noise_scale=2 ** (-args.neg_log_noise_scale),
                disc_bits=args.disc_bits,
                disc_range=args.disc_range,
                ans_mass_bits=args.ans_mass_bits,
                ans_init_seed=0,
                ans_init_bits=(
                    int(np.ceil(total_init_bits_ / args.ans_num_streams)) if total_init_bits_ is not None
                    else args.ans_init_bits  # the --ans_init_bits argument value is the default
                ),
                ans_num_streams=args.ans_num_streams
            )

        if args.mode == 'test':

            results = []
            for i, filename in enumerate(image_list):
                assert args.test_output_filename is not None
                print(f'Processing image {i + 1}/{len(image_list)} - {filename}')
                dataset, bs = load_image_and_crop(filename)

                dataloader, dataset = make_testing_dataloader(
                    dataset, seed=args.seed, limit_dataset_size=args.limit_dataset_size, bs=1
                )

                output = {
                    'args': vars(args),
                    'results': main_compression_test(
                        stream=_make_stream(), model=model, dataloader=dataloader, device=device,
                    )
                }
                with open(os.path.expanduser(args.test_output_filename), 'w') as f:
                    f.write(json.dumps(output) + '\n')

        elif args.mode == 'timing_test_compositional':
            results = []
            for i, filename in enumerate(image_list):
                print('-------------------------------------------------------------------------------------')
                print(f'Processing image {i + 1}/{len(image_list)} - {filename}')
                dataset, bs = load_image_and_crop(filename)

                dataloader, dataset = make_testing_dataloader(
                    dataset, seed=args.seed, limit_dataset_size=args.limit_dataset_size, bs=128
                )

                batches = []
                for (x_raw,) in dataloader:
                    batches.append(x_raw)
                main_timing_test_custom(batches, model=model, stream=_make_stream(), device=device)


if __name__ == '__main__':
    main()
