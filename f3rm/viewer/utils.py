import base64
from io import BytesIO
from typing import Dict, List

import numpy as np
import pandas as pd
import png
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from PIL import Image

CAMERA_TYPES = {
    "PerspectiveCamera": CameraType.PERSPECTIVE,
    "FisheyeCamera": CameraType.FISHEYE,
    "EquirectangularCamera": CameraType.EQUIRECTANGULAR,
}


def get_colormap(colormap, clip=None, gain=1.0, normalize=False):
    """
    https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    get color map from matplotlib
    returns color_map function with signature (x, mask=None),
    where mask is the mask-in for the colormap.

    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)

    def map_color(x, mask=None):
        if clip is not None:
            x = x.clip(*clip)

        if gain is not None:
            x *= gain

        if normalize:
            if mask is None or mask.sum() == 0:
                min, max = x.min(), x.max()
            else:
                min, max = x[mask].min(), x[mask].max()

            x -= min
            x /= max - min + 1e-6
            x[x < 0] = 0

        return cmap(x)

    return map_color


def rotation_matrix(x, y, z, order="xyz"):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(x)
    s1 = np.sin(x)
    c2 = np.cos(y)
    s2 = np.sin(y)
    c3 = np.cos(z)
    s3 = np.sin(z)

    if order == "xzx":
        matrix = np.array(
            [
                [c2, -c3 * s2, s2 * s3],
                [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "xyx":
        matrix = np.array(
            [
                [c2, s2 * s3, c3 * s2],
                [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yxy":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                [s2 * s3, c2, -c3 * s2],
                [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yzy":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                [c3 * s2, c2, s2 * s3],
                [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "zyz":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                [-c3 * s2, s2 * s3, c2],
            ]
        )
    elif order == "zxz":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                [s2 * s3, c3 * s2, c2],
            ]
        )
    elif order == "xyz":
        matrix = np.array(
            [
                [c2 * c3, -c2 * s3, s2],
                [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
            ]
        )
    elif order == "xzy":
        matrix = np.array(
            [
                [c2 * c3, -s2, c2 * s3],
                [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3],
            ]
        )
    elif order == "yxz":
        matrix = np.array(
            [
                [c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                [c2 * s3, c2 * c3, -s2],
                [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2],
            ]
        )
    elif order == "yzx":
        matrix = np.array(
            [
                [c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                [s2, c2 * c3, -c2 * s3],
                [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3],
            ]
        )
    elif order == "zyx":
        matrix = np.array(
            [
                [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                [-s2, c2 * s3, c2 * c3],
            ]
        )
    elif order == "zxy":
        matrix = np.array(
            [
                [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                [-c2 * s3, s2, c2 * c3],
            ]
        )

    return matrix


def focal_len(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


def get_camera_from_three(poses: List[Dict], width: int, height: int) -> Cameras:
    """Get camera object from three.js camera objects"""
    df = pd.DataFrame(poses)

    aspect = df["aspect"].values[0]
    ctype = df["type"].values[0]
    cmatrix = torch.FloatTensor(np.stack(df["matrix"].values)).reshape(-1, 4, 4)
    cmatrix.transpose_(1, 2)
    fov = torch.tensor(df["fov"].values)
    c2w = torch.stack(
        [
            cmatrix[:, 0],
            cmatrix[:, 2],
            cmatrix[:, 1],
            cmatrix[:, 3],
        ],
        dim=1,
    )
    c2w = c2w[:, :3, :]
    c2w = torch.stack([c2w[:, 0], c2w[:, 2], c2w[:, 1]], dim=1)

    return Cameras(
        fx=focal_len(fov, height),
        fy=focal_len(fov, height),
        cx=width / 2,
        cy=width / 2 / aspect,
        camera_to_worlds=c2w,
        camera_type=CAMERA_TYPES[ctype],
    )


def b64jpg(image, quality=90):
    """
    base64 encode the image into a string, using JPEG encoding

    Does not support transparency.
    """
    # remove alpha channel
    if len(image.shape) == 3:
        image = image[:, :, :3]

    buff = BytesIO()
    rgb_pil = Image.fromarray((image * 255).astype(np.uint8))
    rgb_pil.save(buff, format="JPEG", quality=quality)
    img64 = base64.b64encode(buff.getbuffer().tobytes()).decode("utf-8")
    return img64


def b64png(image):
    """
    base64 encode the image into a string, using PNG encoding
    """
    # supports alpha channel

    buff = BytesIO()
    rgb_pil = Image.fromarray((image * 255).astype(np.uint8))
    rgb_pil.save(buff, format="PNG")
    img64 = base64.b64encode(buff.getbuffer().tobytes()).decode("utf-8")
    return img64


def b64png_depth(depth):
    # base64 encode depth map into a string

    buff = BytesIO()

    depth *= 32767
    depth[depth > 65535] = 65535
    im_uint16 = np.round(depth).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(depth.shape[1], depth.shape[0], greyscale=True, bitdepth=16)
    w_depth.write(buff, np.reshape(im_uint16, (-1, depth.shape[1])))

    img64 = base64.b64encode(buff.getbuffer().tobytes()).decode("utf-8")
    return img64


def decode_b64png(b64: str) -> Image:
    """
    Decode a base64 encoded PNG image into a numpy array.
    """
    b64 = b64.split(",")[-1]
    buff = BytesIO(base64.b64decode(b64))
    img = Image.open(buff)
    return img
