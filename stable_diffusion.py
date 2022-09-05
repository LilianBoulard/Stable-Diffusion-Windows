"""
Interface script for Stable Diffusion.

See README.
"""

import base64
import requests

from json import JSONEncoder
from urllib.parse import quote_plus
from argparse import ArgumentParser


if __name__ == "__main__":
    _parser = ArgumentParser('Stable Diffusion interface script.')

    # Copies the arguments that can be found on the Replicate page linked above
    _parser.add_argument(
        '--prompt', type=str, nargs=1, required=True, help='Input prompt',
    )
    _parser.add_argument(
        '--width', type=int, nargs=1, default=512,
        help='Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits',
    )
    _parser.add_argument(
        '--height', type=int, nargs=1, default=512,
        help='Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits',
    )
    _parser.add_argument(
        '--init_image', type=str, nargs=1, required=False,
        help='Inital image to generate variations of. Will be resized to the specified width and height',
    )
    _parser.add_argument(
        '--mask', type=str, nargs=1, required=False,
        help='Black and white image to use as mask for inpainting over init_image. Black pixels are inpainted and white pixels are preserved. Experimental feature, tends to work better with prompt strength of 0.5-0.7',
    )
    _parser.add_argument(
        '--prompt_strength', type=float, nargs=1, default=0.8,
        help='Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image',
    )
    _parser.add_argument(
        '--num_outputs', type=int, nargs=1, default=1,
        help='Number of images to output',
    )
    _parser.add_argument(
        '--num_inference_steps', type=int, nargs=1, default=50,
        help='Number of denoising steps',
    )
    _parser.add_argument(
        '--guidance_scale', type=float, nargs=1, default=7.5,
        help='Scale for classifier-free guidance',
    )
    _parser.add_argument(
        '--seed', type=int, nargs=1, required=False,
        help='Random seed. Leave blank to randomize the seed',
    )

    _args = _parser.parse_args()

    # Check that values are consistent
    # TODO: width, height
    assert 1 <= _args.num_inference_steps <= 500
    assert 1 <= _args.guidance_scale <= 20

    flattened_args = {
        arg: _args.__getattribute__(arg)[0] if isinstance(_args.__getattribute__(arg), list) else _args.__getattribute__(arg)
        for arg in [
            'prompt',
            'width',
            'height',
            'init_image',
            'mask',
            'prompt_strength',
            'num_outputs',
            'num_inference_steps',
            'guidance_scale',
            'seed',
        ]
    }

    assert flattened_args['prompt'] != '', 'Require a prompt !'

    req_args = {
        arg: value
        for arg, value in flattened_args.items()
        if value is not None and value != ''
    }

    data = requests.post(
        url='http://localhost:5000/predictions',
        headers={'Content-Type': 'application/json'},
        data=JSONEncoder().encode({'input': req_args})
    ).json()

    if data["status"] == "succeeded":
        for i, image_data in enumerate(data['output']):
            image_info, image_content = image_data.split(',')
            byte_data = base64.b64decode(image_content)
            mime, _ = image_info.split(';')
            _, extension = mime.split('/')
            with open(f'{i}-{quote_plus(_args.prompt[0])}.{extension}', 'wb') as fl:
                fl.write(byte_data)
