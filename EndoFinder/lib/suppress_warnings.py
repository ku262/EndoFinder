
import warnings

# This relates to Classy Vision transforms that we don't use.
warnings.filterwarnings("ignore", module=".*_(functional|transforms)_video")
# Upstream Classy Vision issue; fix hasn't reached released package.
# https://github.com/facebookresearch/ClassyVision/pull/770
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
# Lightning non-issue (warning false positive).
warnings.filterwarnings("ignore", message=".*overridden after .* initialization.*")
