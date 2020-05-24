import arrow
import os
import requests
import xmltodict
from urllib.request import urlopen

from src.meteoApi.radarTilesDownloader import RadarTilesDownloader


class MeteoDataDownloader:
    baseDir = ''
    stepMinutes = 15
    imagesBeforeCount = 8

    @staticmethod
    def get_newest_available_time_from_meteo():
        r = requests.get('http://www.meteo.lt/mapsfree?SERVICE=WMS&REQUEST=GetCapabilities', headers={
          'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36",
          'Host': "www.meteo.lt",
          'Cookie': "Cookie: GUEST_LANGUAGE_ID=lt_LT; COOKIE_SUPPORT=true; _ga=GA1.2.1171208369.1527599128; JSESSIONID=49AB0D1410388A27A6E5A620628DB6D0; _gid=GA1.2.9622810.1590303177; _gat=1; LFR_SESSION_STATE_20158=1590312230006"
        })
        data = xmltodict.parse(r.content)
        x = data['WMS_Capabilities']['Capability']['Layer']['Layer']['Layer']
        radar_layer = None
        for layer in x:
            if layer['Name'] == 'NP2_radar':
                radar_layer = layer
        return radar_layer['Dimension']['@default']

    def get_current_time(self):
        utc = arrow.utcnow()
        time = utc.time()
        minutes = time.minute
        intervals = minutes // self.stepMinutes
        rounded_minutes = self.stepMinutes * intervals
        utc = utc.replace(minute=rounded_minutes, second=0)
        return utc

    def set_base_dir(self, dir):
        self.baseDir = dir
        return self

    def set_step_minutes(self, minutes):
        self.stepMinutes = minutes
        return self

    def set_images_count(self, count):
        self.imagesBeforeCount = count
        return self

    def load_radar_data(self):
        os.makedirs(os.path.join(self.baseDir, 'actual'), exist_ok=True)
        newest_time = self.get_newest_available_time_from_meteo()
        time = arrow.get(newest_time)
        downloader = RadarTilesDownloader()
        downloader.set_dir(self.baseDir)
        files = []
        for i in range(0, self.imagesBeforeCount):
            print('loading image', str(i))
            file_name = downloader.load_and_save_radar_image(time)
            files.append(file_name)
            time = time.shift(minutes=-1*self.stepMinutes)
        return dict(
         source_time=newest_time,
         files=files
        )
