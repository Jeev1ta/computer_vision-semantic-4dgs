#!/usr/bin/env bash
set -e
echo "Running environment check..."
python3 -c "
import sys, numpy, plyfile, cv2, PIL, matplotlib, yaml
print('Python:', sys.version.split()[0])
print('NumPy:', numpy.__version__)
print('Plyfile:', plyfile.__version__)
print('OpenCV:', cv2.__version__)
print('Pillow:', PIL.__version__)
print('Matplotlib:', matplotlib.__version__)
print('PyYAML:', yaml.__version__)
"
