import os
from jax_fem.common import make_video

data_path = os.path.join(os.path.dirname(__file__), 'data')
make_video(data_path)