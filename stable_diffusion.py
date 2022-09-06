"""
Interface script for Stable Diffusion.

Refer to the original blog post for instructions:
https://lilian-boulard.notion.site/Self-host-Stable-Diffusion-on-Windows-31f7c407d2bd47d4b5ff389406dc27a3
and to the project's repository for more information:
https://github.com/LilianBoulard/Stable-Diffusion-Windows
"""

import base64
import requests

from json import JSONEncoder
from datetime import datetime
from urllib.parse import quote_plus
from argparse import ArgumentParser


def now() -> str:
    _now = datetime.now()
    return f"{_now.year}-{_now.month}-{_now.day} {_now.hour}:{_now.minute}:{_now.second}"


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

    _parser.add_argument(
        '--times', type=int, nargs=1, required=False, default=1,
        help='How many images we will generate using these parameters. '
             'Note: if "seed" is also passed, this will only generate one. '
    )

    _args = _parser.parse_args()

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
            'times',
        ]
    }

    # Check that values are consistent
    assert (
        (flattened_args['height'] <= 1024 and flattened_args['width'] <= 768)
        or
        (flattened_args['height'] <= 768 and flattened_args['width'] <= 1024)
    )
    assert flattened_args['prompt'] != '', 'Require a prompt !'
    assert 1 <= flattened_args['num_inference_steps'] <= 500
    assert 1 <= flattened_args['guidance_scale'] <= 20
    assert flattened_args['num_outputs'] in [1, 4]
    if flattened_args['seed'] is not None and flattened_args['times'] != 1:
        print(f'A seed ({flattened_args["seed"]} was passed, '
              f'but {flattened_args["times"]} images were requested. '
              f'As this would generate the same image multiple times, '
              f'"times" will be ignored and a single image will be generated. ')
        flattened_args['times'] = 1

    req_args = {
        arg: value
        for arg, value in flattened_args.items()
        if value is not None and value != ''
    }

    url = 'http://localhost:5000/predictions'
    for _ in range(flattened_args['times']):
        try:
            data = requests.post(
                url=url,
                headers={'Content-Type': 'application/json'},
                data=JSONEncoder().encode({'input': req_args})
            ).json()
        except requests.exceptions.ConnectionError:
            print(f'Could not connect to the server ({url!r}). '
                  f'Please ensure the docker container is up and running. ')
        else:
            # Extract the image information and save them in a file.
            if "status" in data and data["status"] == "succeeded":
                for i, image_data in enumerate(data['output']):
                    image_info, image_content = image_data.split(',')
                    byte_data = base64.b64decode(image_content)
                    mime, _ = image_info.split(';')
                    _, extension = mime.split('/')
                    with open(f'{quote_plus(_args.prompt[0])} {now()}.{extension}', 'wb') as fl:
                        fl.write(byte_data)
            else:
                print("Error:", data)
