# Native optical flow runtime

This module owns dense local optical-flow inference between equal-size images.
On Apple platforms it uses the system Vision framework and writes standard
Middlebury `.flo` vectors with typed JSON metadata for downstream VFX tools.
