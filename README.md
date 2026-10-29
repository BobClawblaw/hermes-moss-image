# hermes-moss-image

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) image-generation plugin for
[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) served by
[TensorFold](https://github.com/ashhart/TensorFold) (`POST /v1/images/generations`).

Hermes's image tool passes only a prompt and an aspect ratio. This plugin reads the options MOSS-Image takes out of
the prompt and sends them to the server:

| In the prompt | Sent |
| --- | --- |
| `512x512`, `1344 x 768`, `768x1344px` | that size, each side rounded to a multiple of 16 |
| `30 steps` | 30 denoising steps (1 to 200) |
| `seed 7` | that seed: the same prompt and seed give the same picture |
| `guidance 4` or `cfg 4` | that guidance strength |
| no size | the aspect ratio's size: 1024x1024, 1536x1024 or 1024x1536 |

The options are removed from the prompt before it is sent. A size past the server's limits (1536 a side, 1536 x
1024's pixels) is refused with a message naming them, instead of being shrunk silently.

## Requirements

- Hermes Agent 0.22 (tested on 0.22.5).
- A TensorFold server serving MOSS-Image-Base. The image route is part of TensorFold's MOSS-Image work, which is
  under review upstream and not yet in a release; until it lands, the server comes from that work's branch.

## Install

```sh
cp -r moss_image ~/.hermes/plugins/
```

Then in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - moss-image
image_gen:
  provider: moss
  moss:
    base_url: http://127.0.0.1:8899/v1   # the TensorFold server
    model: moss-image-base               # the server's --name
    # key_env: MOSS_IMAGE_KEY            # only if the server was started with --api-key-file
```

Restart the gateway (`hermes gateway restart`). Desktop sessions also need the image toolset enabled for their
platform (`platform_toolsets: desktop: [hermes-desktop, image_gen]`).

## Use

Ask Hermes for a picture and put any options in the request:

> make a 768x512 picture of a paper boat on a puddle after rain, 30 steps, seed 3

Pictures are saved to Hermes's image cache (`~/.hermes/cache/generated/images/moss_<size>_<time>.png`).
MOSS-Image makes new pictures only: requests with source or reference images are refused.

## License

MIT
