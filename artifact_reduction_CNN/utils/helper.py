import SimpleITK as sitk
import numpy


def save_as_itk(data, file_name, cf):
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(cf.spacing)
    image.SetOrigin(cf.origin)
    image.SetDirection(cf.direction)
    sitk.WriteImage(image, file_name)


def crop3D_mid(data, size):
    [l, m, n] = data.shape
    [out_l, out_m, out_n] = size
    start_l = (l - out_l) // 2
    start_m = (m - out_m) // 2
    start_n = (n - out_n) // 2
    res = data[start_l:start_l+out_l, start_m:start_m+out_m, start_n:start_n+out_n]
    return res


def crop3D_luCorner(data, size):
    [out_l, out_m, out_n] = size
    res = data[:out_l, :out_m, :out_n]
    return res