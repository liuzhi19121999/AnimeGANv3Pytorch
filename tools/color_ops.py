"""Color Space Ops."""

import torch
from torch import Tensor

def rgb_to_bgr(input: Tensor, name=None):
    """
    Convert a RGB image to BGR.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    rgb = torch.unbind(input, dim=-1)
    r, g, b = rgb[0], rgb[1], rgb[2]
    return torch.stack([b, g, r], dim=-1)


def bgr_to_rgb(input: Tensor, name=None):
    """
    Convert a BGR image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    bgr = torch.unbind(input, dim=-1)
    b, g, r = bgr[0], bgr[1], bgr[2]
    return torch.stack([r, g, b], dim=-1)


def rgb_to_rgba(input: Tensor, name=None):
    """
    Convert a RGB image to RGBA.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 4]`) or 4-D (`[N, H, W, 4]`) Tensor.
    """
    rgb = torch.unbind(input, dim=-1)
    r, g, b = rgb[0], rgb[1], rgb[2]
    a = torch.zeros_like(r)
    return torch.stack([r, g, b, a], dim=-1)


def rgba_to_rgb(input: Tensor, name=None):
    """
    Convert a RGBA image to RGB.
    Args:
      input: A 3-D (`[H, W, 4]`) or 4-D (`[N, H, W, 4]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    rgba = torch.unbind(input, dim=-1)
    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]
    return torch.stack([r, g, b], dim=-1)


def rgb_to_ycbcr(input: Tensor, name=None):
    """
    Convert a RGB image to YCbCr.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = torch.tensor(input)

    assert input.dtype == torch.uint8
    value = input.to(dtype=torch.float32, copy=True)
    value = value / 255.0
    value = rgb_to_ypbpr(value)
    value = value * torch.tensor([219, 224, 224], value.dtype)
    value = value + torch.tensor([16, 128, 128], value.dtype)
    return value.to(dtype=input.dtype)


def ycbcr_to_rgb(input: Tensor, name=None):
    """
    Convert a YCbCr image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = torch.tensor(input)

    assert input.dtype == torch.uint8
    value = input.to(dtype=torch.float32, copy=True)
    value = value - torch.tensor([16, 128, 128], value.dtype)
    value = value / torch.tensor([219, 224, 224], value.dtype)
    value = ypbpr_to_rgb(value)
    value = value * 255.0
    return value.to(dtype=input.dtype)


def rgb_to_ypbpr(input: Tensor, name=None):
    """
    Convert a RGB image to YPbPr.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = torch.tensor(input)
    assert input.dtype in (torch.float16, torch.float32, torch.float64)

    kernel = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        input.dtype,
    )

    kernel_transposed = kernel.transpose(0, -1)

    return torch.tensordot(input, kernel_transposed, dims=[-1, 0])


def ypbpr_to_rgb(input: Tensor, name=None):
    """
    Convert a YPbPr image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = torch.tensor(input)
    assert input.dtype in (torch.float16, torch.float32, torch.float64)

    # inv of:
    # [[ 0.299   , 0.587   , 0.114   ],
    #  [-0.168736,-0.331264, 0.5     ],
    #  [ 0.5     ,-0.418688,-0.081312]]
    kernel = torch.tensor(
        [
            [1.00000000e00, -1.21889419e-06, 1.40199959e00],
            [1.00000000e00, -3.44135678e-01, -7.14136156e-01],
            [1.00000000e00, 1.77200007e00, 4.06298063e-07],
        ],
        input.dtype,
    )

    kernel_transposed = kernel.transpose(0, -1)

    return torch.tensordot(input, kernel_transposed, dims=[-1, 0])

    # return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))


def rgb_to_ydbdr(input: Tensor, name=None):
    """
    Convert a RGB image to YDbDr.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = torch.tensor(input)
    assert input.dtype in (torch.float16, torch.float32, torch.float64)

    kernel = torch.tensor(
        [[0.299, 0.587, 0.114], [-0.45, -0.883, 1.333], [-1.333, 1.116, 0.217]],
        input.dtype,
    )

    kernel_transposed = kernel.transpose(0, -1)

    return torch.tensordot(input, kernel_transposed, dims=[-1, 0])

    # return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))


def ydbdr_to_rgb(input: Tensor, name=None):
    """
    Convert a YDbDr image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = torch.tensor(input)
    assert input.dtype in (torch.float16, torch.float32, torch.float64)

    # inv of:
    # [[    0.299,   0.587,    0.114],
    #  [   -0.45 ,  -0.883,    1.333],
    #  [   -1.333,   1.116,    0.217]]
    kernel = torch.tensor(
        [
            [1.00000000e00, 9.23037161e-05, -5.25912631e-01],
            [1.00000000e00, -1.29132899e-01, 2.67899328e-01],
            [1.00000000e00, 6.64679060e-01, -7.92025435e-05],
        ],
        input.dtype,
    )

    kernel_transposed = kernel.transpose(0, -1)

    return torch.tensordot(input, kernel_transposed, dims=[-1, 0])

    # return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))


def rgb_to_hsv_torch(rgb: Tensor):  
    # 确保输入是 torch.FloatTensor，并且值在 [0, 1] 范围内  
    rgb = rgb.clamp(0, 1)
    # RGB 到 HSV 转换  
    # 计算 V (明度)  
    V = rgb.max(dim=-1, keepdim=True)[0]  
    # 计算 S (饱和度)  
    S = torch.where(V != 0, (V - rgb) / V, torch.zeros_like(V))  
    # 计算 H (色调)  
    # 定义一个 delta，以避免除以零  
    delta = V - rgb.min(dim=-1, keepdim=True)[0]  
    # 初始化 H  
    H = torch.zeros_like(V)  
    # 计算 R, G, B 对应的 H  
    H[rgb[:, :, 0] == V] = (60 * (rgb[:, :, 1] - rgb[:, :, 2]) / delta)[rgb[:, :, 0] == V]  
    H[rgb[:, :, 1] == V] = 2 + (60 * (rgb[:, :, 2] - rgb[:, :, 0]) / delta)[rgb[:, :, 1] == V]  
    H[rgb[:, :, 2] == V] = 4 + (60 * (rgb[:, :, 0] - rgb[:, :, 1]) / delta)[rgb[:, :, 2] == V]  
    # 处理 0 度角和 360 度角的情况  
    H[delta == 0] = 0  
    H = (H / 60) % 1
    # 将 H, S, V 合并为一个张量  
    hsv = torch.cat((H, S, V), dim=-1)  
    return hsv  

def rgb_to_hsv(input: Tensor, name=None):
    """
    Convert a RGB image to HSV.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.rgb_to_hsv for completeness
    input = torch.tensor(input)
    return rgb_to_hsv_torch(input)

def hsv_to_rgb_torch(hsv: Tensor):  
    # 确保输入是 torch.FloatTensor，并且值在 [0, 1] 范围内  
    hsv = hsv.clamp(0, 1)  
      
    # 分离出 h, s, v 三个通道  
    h, s, v = hsv.unbind(-1)  
      
    # H 通道的值域是 [0, 1]，转换为 [0, 360]  
    h = h * 360  
      
    # 初始化 RGB 通道  
    r = torch.zeros_like(h)  
    g = torch.zeros_like(h)  
    b = torch.zeros_like(h)  
      
    # 根据 HSV 到 RGB 的转换公式进行转换  
    # 注意：这里使用 torch.where 来处理区间边界的情况  
    c = v * s  
    x = c * (1 - torch.abs(h.fmod(60*2) - 60) / 60)  
      
    indices = torch.floor(h / 60)  
      
    for i in range(6):  
        cond = indices == i  
        r[cond] = torch.where(h[cond] < 60*(i+1), c[cond], x[cond])  
        g[cond] = torch.where(i == 1, x[cond], torch.where(i == 2, c[cond], torch.where(h[cond] < 60*(i+3), x[cond], torch.zeros_like(x[cond]))))  
        b[cond] = torch.where(indices == 0, torch.zeros_like(x[cond]), torch.where(indices == 1 or indices == 5, x[cond], torch.where(i == 2, c[cond], torch.zeros_like(x[cond]))))  
      
    # 将 RGB 三个通道合并  
    m = v - c  
    rgb = torch.stack((r + m, g + m, b + m), dim=-1)  
      
    # 确保值在 [0, 1] 范围内  
    rgb = rgb.clamp(0, 1)  
      
    return rgb

def hsv_to_rgb(input: Tensor, name=None):
    """
    Convert a HSV image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.hsv_to_rgb for completeness
    input = torch.tensor(input)
    return hsv_to_rgb_torch(input)

def rgb_to_yiq_torch(rgb_image: Tensor):  
    # 定义RGB到YIQ的转换矩阵  
    # 假设RGB值在[0, 1]范围内，且已经转换为float32  
    # TensorFlow中的矩阵可能略有不同，但这里是常见的转换矩阵  
    rgb_to_yiq_matrix = torch.tensor([[0.299, 0.587, 0.114],  # Y  
                                      [0.595716, -0.274453, -0.321263],  # I  
                                      [0.211456, -0.522591, 0.311135]], dtype=torch.float32)  # Q  
      
    # 如果图像是batch的，我们需要增加一个维度来匹配矩阵的维度  
    # 例如，如果rgb_image的形状是[batch_size, height, width, 3]，我们需要将其reshape为[batch_size, height * width, 3]  
    # 但为了简单起见，这里我们假设rgb_image是一个单独的图像，即形状为[height, width, 3]  
    # 如果需要处理batch，请添加相应的reshape操作
    rgb_image_data = rgb_image.clone()
    if rgb_image.shape.__len__() == 4:
        rgb_image_data = rgb_image.reshape(
            (rgb_image_data.shape[0], rgb_image_data.shape[1] * rgb_image_data.shape[2], rgb_image_data.shape[3]))
      
    # 使用矩阵乘法进行转换  
    yiq_image = torch.matmul(rgb_image.permute(2, 0, 1), rgb_to_yiq_matrix)
    yiq_image = yiq_image.permute(1, 2, 0)

    if rgb_image.shape.__len__() == 4:
        rgb_image_data = yiq_image.reshape(
            (rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2], rgb_image.shape[3]))
    # 将结果重新排列回[height, width, 3]的形状（如果需要的话）
    # 注意：如果原始图像是uint8类型，并且我们希望在PyTorch中也保持相同的范围，  
    # 我们可能需要将结果转换回[0, 255]的整数范围，但这通常不是必需的，因为深度学习模型通常处理浮点数据  
    return yiq_image

def rgb_to_yiq(input: Tensor, name=None):
    """
    Convert a RGB image to YIQ.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.rgb_to_yiq for completeness
    input = torch.tensor(input)
    return rgb_to_yiq_torch(input)

def yiq_to_rgb_torch(yiq_image):  
    # 定义YIQ到RGB的转换矩阵  
    # 注意：这里的矩阵可能需要根据实际使用的YIQ到RGB转换公式进行调整
    yiq_img_data = torch.tensor(yiq_image)

    yiq_to_rgb_matrix = torch.tensor([[1.0, 0.956, 0.621],  
                                      [1.0, -0.272, -0.647],  
                                      [1.0, -1.107, 1.705]], dtype=torch.float32)  
  
    # 假设yiq_image的形状是[height, width, 3]，且数据类型为torch.float32  
    # 如果不是，可能需要进行相应的调整  
  
    # 使用矩阵乘法进行转换  
    # 注意：这里假设最后一个维度是颜色通道（YIQ），所以需要进行转置  
    rgb_image = torch.matmul(yiq_to_rgb_matrix, yiq_img_data.permute(2, 0, 1)).permute(1, 2, 0)  
  
    # 注意：这里可能需要进一步的调整，例如裁剪到[0, 1]范围或转换为uint8等，具体取决于你的应用场景  
  
    return rgb_image

def yiq_to_rgb(input: Tensor, name=None):
    """
    Convert a YIQ image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.yiq_to_rgb for completeness
    input = torch.tensor(input)
    return yiq_to_rgb_torch(input)

def rgb_to_yuv_torch(rgb_image):  
    # 确保输入是torch.FloatTensor类型，并且范围在[0, 1]  
    # rgb_image = rgb_image.float()  
  
    # 标准化RGB到[0, 1]  
    # 这一步可能在之前已经完成，但为了确保，我们再次执行  
    rgb_img_data = torch.tensor(rgb_image)
    rgb_img_data = rgb_img_data / 255.0  
  
    # RGB to YUV 转换公式（这里使用了简化的近似版本）  
    # Y = 0.299 * R + 0.587 * G + 0.114 * B  
    # U = -0.14713 * R - 0.28886 * G + 0.436 * B + 128  
    # V = 0.615 * R - 0.51498 * G - 0.10001 * B + 128  
    # 注意：这里的+128是为了将UV值转换到[0, 255]范围，但PyTorch通常使用[0, 1]范围，所以这里可以省略  
  
    R, G, B = rgb_img_data[:, :, 0], rgb_img_data[:, :, 1], rgb_img_data[:, :, 2]  
  
    Y = 0.299 * R + 0.587 * G + 0.114 * B  
    U = -0.14713 * R - 0.28886 * G + 0.436 * B  
    V = 0.615 * R - 0.51498 * G - 0.10001 * B  
  
    # 将YUV堆叠在一起，注意这里YUV的顺序可能与某些标准不同  
    yuv_image = torch.stack((Y, U, V), dim=2)  
  
    # 如果需要，可以将YUV值缩放到[0, 1]范围  
    yuv_image = yuv_image / 255.0  # 但通常YUV值在[0, 255]或[16, 235]等范围内  
  
    return yuv_image

def rgb_to_yuv(input: Tensor, name=None):
    """
    Convert a RGB image to YUV.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.rgb_to_yuv for completeness
    input0 = torch.tensor(input)
    return rgb_to_yuv_torch(input0)

def yuv_to_rgb_torch(yuv_image):  
    # 假设 yuv_image 的形状是 [height, width, 3]，且数据类型为 torch.float32  
    # YUV 到 RGB 的转换公式（简化的近似版本）  
    # R = Y + 1.13983 * (V - 128)  
    # G = Y - 0.39465 * (U - 128) - 0.58060 * (V - 128)  
    # B = Y + 2.03211 * (U - 128)  
    # 注意：这里的+128和-128是为了将UV值从[0, 1]或[0, 255]转换到[-128, 127]范围，但在PyTorch中我们通常使用[0, 1]范围，所以这里需要调整  
  
    Y, U, V = yuv_image[:, :, 0], yuv_image[:, :, 1], yuv_image[:, :, 2]  
  
    R = Y + 1.13983 * (V - 0.5)  # 假设 V 已经在 [0, 1] 范围  
    G = Y - 0.39465 * (U - 0.5) - 0.58060 * (V - 0.5)  
    B = Y + 2.03211 * (U - 0.5)  
  
    # 将 RGB 堆叠在一起  
    rgb_image = torch.stack((R, G, B), dim=2)  
  
    # 确保 RGB 值在 [0, 1] 范围内  
    rgb_image = torch.clamp(rgb_image, 0, 1)  
  
    return rgb_image

def yuv_to_rgb(input: Tensor, name=None):
    """
    Convert a YUV image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.yuv_to_rgb for completeness
    input = torch.tensor(input)
    return yuv_to_rgb_torch(input)


def rgb_to_xyz(input: Tensor, name=None):
    """
    Convert a RGB image to CIE XYZ.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # input = torch.tensor(input)
    assert input.dtype in (torch.float16, torch.float32, torch.float64)

    kernel = torch.tensor(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        dtype=input.dtype,
    )
    value = torch.where(
        input > 0.04045,
        torch.pow((input + 0.055) / 1.055, 2.4),
        input / 12.92,
    )

    kernel_transposed = kernel.transpose(0, -1)

    return torch.matmul(value, kernel_transposed)
    
    # return tf.tensordot(value, tf.transpose(kernel), axes=((-1,), (0,)))


def xyz_to_rgb(input: Tensor, name=None):
    """
    Convert a CIE XYZ image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input0 = torch.tensor(input)
    assert input0.dtype in (torch.float16, torch.float32, torch.float64)

    # inv of:
    # [[0.412453, 0.35758 , 0.180423],
    #  [0.212671, 0.71516 , 0.072169],
    #  [0.019334, 0.119193, 0.950227]]
    kernel = torch.tensor(
        [
            [3.24048134, -1.53715152, -0.49853633],
            [-0.96925495, 1.87599, 0.04155593],
            [0.05564664, -0.20404134, 1.05731107],
        ],
        input0.dtype,
    )

    kernel_transposed = kernel.transpose(0, -1)
    value = torch.matmul(input, kernel_transposed)
    value = torch.where(
        torch.gt(value, torch.tensor(0.0031308)),
        torch.math.pow(value, 1.0 / 2.4) * 1.055 - 0.055,
        value * 12.92,
    )
    return torch.clamp(value, 0, 1)


def rgb_to_lab(input: Tensor, illuminant="D65", observer="2", name=None):
    """
    Convert a RGB image to CIE LAB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
      observer : {"2", "10"}, optional
        The aperture angle of the observer.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # input = torch.tensor(input)
    assert input.dtype in (torch.float16, torch.float32, torch.float64)

    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = torch.tensor(illuminants[illuminant.upper()][observer], dtype=input.dtype)

    xyz = rgb_to_xyz(input)

    xyz = xyz / coords

    xyz = torch.where(
        torch.gt(xyz, torch.tensor(0.008856)),
        torch.pow(xyz, 1.0 / 3.0),
        xyz * 7.787 + 16.0 / 116.0,
    )

    xyz = torch.unbind(xyz, dim=-1)
    x, y, z = xyz[0], xyz[1], xyz[2]

    # Vector scaling
    l = (y * 116.0) - 16.0
    a = (x - y) * 500.0
    b = (y - z) * 200.0

    return torch.stack([l, a, b], dim=-1)


def lab_to_rgb(input: Tensor, illuminant="D65", observer="2", name=None):
    """
    Convert a CIE LAB image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
      observer : {"2", "10"}, optional
        The aperture angle of the observer.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input0 = torch.tensor(input)
    assert input0.dtype in (torch.float16, torch.float32, torch.float64)

    lab = input0
    lab = torch.unbind(lab, dim=-1)
    l, a, b = lab[0], lab[1], lab[2]

    y = (l + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    z = torch.clamp_min(z, 0)

    xyz = torch.stack([x, y, z], axis=-1)

    xyz = torch.where(
        torch.gt(xyz, torch.tensor(0.2068966)),
        torch.math.pow(xyz, 3.0),
        (xyz - 16.0 / 116.0) / 7.787,
    )

    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = torch.tensor(illuminants[illuminant.upper()][observer], input0.dtype)

    xyz = xyz * coords

    return xyz_to_rgb(xyz)


def rgb_to_grayscale(input, name=None):
    """
    Convert a RGB image to Grayscale (ITU-R).
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: This rgb_to_grayscale conforms to skimage.color.rgb2gray
    # and is different from tf.image.rgb_to_grayscale
    input0 = torch.tensor(input).float()

    # value = tf.image.convert_image_dtype(input, tf.float32)
    coeff = torch.tensor([0.2125, 0.7154, 0.0721], dtype=input0.dtype)
    value = torch.tensordot(value, coeff, [-1, -1])
    value = torch.unsqueeze(value, -1)
    return torch.tensor(value, torch.uint8)

if __name__ == '__main__':
    import cv2
    image_foder = './dataset/Hayao/style/11.jpg'
    img = cv2.imread(image_foder)
    print(img.shape)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
    p = torch.tensor(img, dtype=torch.float32)
    x=rgb_to_lab(p)
    print(x)
    y=p[:,:,0]/100
    print(y)
    # cv2.imshow('dd',y.numpy())
    cv2.waitKey(0)
