from os import path

import shutil

from src.meteoApi.imagesConcater import ImagesConcater
import os
import urllib.request
import shutil
import requests
import time as tm

class RadarTilesDownloader:
    baseDir = ''
    workingDir = ''

    startColumn = 71
    endColumn = 73

    startRow = 39
    endRow = 41

    def set_dir(self, dir):
        self.baseDir = dir
        return self

    def download_tile(self, time, x, y, result):
        formatted = time.format('YYYY-MM-DDTHH:mm:ss')
        url = "http://www.meteo.lt/mapsfree?" +\
            "&SERVICE=WMS" +\
            "&VERSION=1.3.0" +\
            "&REQUEST=GetGTile" +\
            "&CRS=EPSG:900913" +\
            "&TILEZOOM=7&TILEROW="+str(y) +\
            "&TILECOL="+str(x)+"&TRANSPARENT=true" +\
            "&DPI=72" +\
            "&LAYER=NP2_radar" +\
            "&TIMESTAMP=0" +\
            "&time="+formatted +\
            "Z&DIM_="
        print(url)

        response = requests.get(url, stream=True, headers={
          'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36",
          'Host': "www.meteo.lt",
          'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
          'Accept-Encoding': 'gzip, deflate',
          'Connection': 'keep-alive',
          'Referer': 'http://www.meteo.lt/lt/radaru-informacija'
        })
        with open(result, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

    def download_tiles(self, time):
        os.makedirs(self.workingDir, exist_ok=True)
        n = 0
        for x in range(self.startColumn, self.endColumn+1):
            for y in range(self.startRow, self.endRow+1):
                result_file = path.join(self.workingDir, str(n)+'.png')
                self.download_tile(time, x, y, result_file)
                n+=1

    def load_and_save_radar_image(self, time):
        concater = ImagesConcater()
        self.workingDir = path.join(self.baseDir, time.format('YYYY-MM-DD--HH-mm-ss'))
        file_name = time.format('YYYY-MM-DD--HH-mm-ss') + '.png'
        result_file = path.join(self.baseDir, 'actual', file_name)
        self.download_tiles(time)
        concater.concat_images(self.workingDir, result_file)
        shutil.rmtree(self.workingDir)
        return file_name
