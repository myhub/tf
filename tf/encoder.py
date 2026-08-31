import torch
import torch.autograd
import torch.nn
import os
from pathlib import Path
import ctypes
import random
import tempfile

_c_float_p = ctypes.POINTER(ctypes.c_float)

def _ptr_float(t):
    if isinstance(t, int):
        return ctypes.cast(t, _c_float_p)
    else:
        assert t.is_contiguous()
        return ctypes.cast(t.data_ptr(), _c_float_p)

_lib = ctypes.CDLL(Path(__file__).parent.joinpath("libtf.so").resolve().__str__())
assert _lib is not None

_lib.tf_open.restype = ctypes.c_int
_lib.tf_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, _c_float_p]

_lib.tf_close.restype = ctypes.c_int
_lib.tf_close.argtypes = [ctypes.c_int]

_lib.tf_forward.restype = ctypes.c_int
_lib.tf_forward.argtypes = [ctypes.c_int, _c_float_p, _c_float_p, _c_float_p, ctypes.c_int]

_lib.tf_backward.restype = ctypes.c_int
_lib.tf_backward.argtypes = [ctypes.c_int, 
    _c_float_p, _c_float_p, _c_float_p, _c_float_p, _c_float_p, _c_float_p, ctypes.c_int]

_lib.tf_renew.restype = ctypes.c_int
_lib.tf_renew.argtypes = [ctypes.c_int, ctypes.c_float]

_lib.tf_save_model.restype = ctypes.c_int
_lib.tf_save_model.argtypes = [ctypes.c_int, ctypes.c_char_p]

_lib.tf_load_model.restype = ctypes.c_int
_lib.tf_load_model.argtypes = [ctypes.c_int, ctypes.c_char_p]

_lib.tf_reset_parameters.restype = ctypes.c_int
_lib.tf_reset_parameters.argtypes = [ctypes.c_int, _c_float_p]

_lib.tf_util_grad.restype = ctypes.c_int
_lib.tf_util_grad.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, (ctypes.c_double*1)]


class _tf_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, batch_size, pid):
        ctx.save_for_backward(x)
        ctx.batch_size = int(batch_size)
        ctx.pid = int(pid)
        y = torch.empty_like(x)
        _lib.tf_forward(ctx.pid, _ptr_float(x), _ptr_float(y), _ptr_float(0), ctx.batch_size)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = torch.empty_like(x)
        _lib.tf_backward(ctx.pid, 
            _ptr_float(x), _ptr_float(dx),
            _ptr_float(0), _ptr_float(dy.contiguous()), 
            _ptr_float(0), _ptr_float(0),
            ctx.batch_size)
        return dx, None, None


def con_dumps(con_obj):
    con_str = "{"   
    for k in con_obj:
        obj = con_obj[k]
        con_str += k + ":"
        if isinstance(obj, (int, float)):    
            con_str += str(con_obj[k])
        else:
            con_str += "'" + str(obj).replace("'", "''") + "'"
        con_str += ","
    con_str += "}"
    return con_str.encode("utf-8")


class TfEncoder(torch.nn.Module):
    def __init__(self, t, c, bmax, depth=4, pid=0, device="cuda", **kwargs):
        super().__init__()

        self.pid = int(pid)
        self.t = int(t)
        self.c = int(c)
        self.bmax = int(bmax)
        self.depth = int(depth)

        self.device = self.get_device(device)
        
        dev_props = torch.cuda.get_device_properties(self.device.index)
        sm_ver = f"sm_{dev_props.major}{dev_props.minor}"
        assert sm_ver in "sm_75/sm_80/sm_86/sm_89/sm_90", sm_ver
        
        self.tf_op = _tf_func.apply

        con_obj = dict(bmax=bmax, t=t, c=c, depth=depth, **kwargs)
        
        _lib.tf_open(self.pid, self.device.index, con_dumps(con_obj), _ptr_float(0))
        # print("self.device.index", self.device.index)

        self._para1 = torch.nn.Parameter(torch.zeros((1, ), dtype=torch.float32, device=self.device))
        
        # 初始化参数
        _lib.tf_reset_parameters(self.pid, _ptr_float(0))
        
    def renew(self, lr):
        with torch.no_grad():
            _lib.tf_renew(self.pid, lr)

    def util_grad(self, sid, dval=0):
        args = (ctypes.c_double*1)(0)
        err = _lib.tf_util_grad(self.pid, sid, dval, args)
        # print(args[0])
        return args[0]

    def save_model(self):                                                                                                                                       
        with torch.no_grad():
            with tempfile.NamedTemporaryFile("wb", delete=True) as f:
                fpath = str(f.name)
                _lib.tf_save_model(self.pid, fpath.encode("utf8"))
                model_data = Path(fpath).read_bytes()
                return model_data

    def load_model(self, model_data):
        with torch.no_grad():
            with tempfile.NamedTemporaryFile("wb", delete=True) as f:
                fpath = str(f.name)
                Path(fpath).write_bytes(model_data)
                _lib.tf_load_model(self.pid, str(fpath).encode("utf8"))

    def __del__(self):
        _lib.tf_close(self.pid)

    @staticmethod
    def get_device(t):
        if hasattr(t, "device"):
            device = t.device
        else:
            device = torch.device(t)

        assert device.type == "cuda"

        if device.index is None: 
            device = torch.device(device.type, 0)
        return device

    def forward(self, x):
        device = self.get_device(self._para1)

        assert device.index == self.device.index
        # if device.index != self.device.index:
        #     _lib.tf_close(self.pid)
        #     _lib.tf_open(self.pid, device.index, _ptr_float(0))

        b, t, c = x.shape
        assert b <= self.bmax
        assert x.is_contiguous()
        # print("x", x.shape, x.device)
        y = self.tf_op(x, b, self.pid)
        return y

    


    
        

        

