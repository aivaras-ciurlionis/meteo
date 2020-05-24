# from PIL import Image
# import requests
# from io import BytesIO
#
# for alg in ['CNN4L', 'AE', 'ConvLSTM']:
#     for i in range(15, 15*15, 15):
#         response = requests.get('https://meteorologydata.blob.core.windows.net/predicted/'+alg+'_TEST_DATE_'+str(i)+'m_.png')
#         img = Image.open(BytesIO(response.content))
#         img.save('output/'+alg+str(i)+'.png')
#
#
from src.prediction.predictionWrapper import PredictionWrapper

PredictionWrapper.predict()