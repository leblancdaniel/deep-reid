from torchreid.utils import FeatureExtractor
import time


extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path='torchreid/models/osnet_x0_25_imagenet.pth',
    device='cpu'
)

image_list = [
    'torchreid/data/img.png'
]

total_time = 0
for i in range(100):
    
    time1 = time.time()
    features = extractor(image_list)
    time2 = time.time()
    total_time += (time2 - time1)

    #print(features.shape) # output (N, 512)

avg_time = total_time / len(range(100))
print(avg_time, "Avg time per loop")
print( 1 / avg_time, "Avg FPS")
    