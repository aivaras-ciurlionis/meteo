from src.utilities.fileProcessing.ImageSequencesLoader import ImageSequencesLoader

sequencesLoader = ImageSequencesLoader()
imageSequences = sequencesLoader.select_folder('../pics').load_sequences()
print(imageSequences[0])
print(len(imageSequences))