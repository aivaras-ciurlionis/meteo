from src.meteoApi.meteoDataDownloader import MeteoDataDownloader

meteo = MeteoDataDownloader()
meteo.set_base_dir('../meteo-out').set_images_count(10).load_radar_data()
