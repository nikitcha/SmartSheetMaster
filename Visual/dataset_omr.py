#%% HOMUS
from omrdatasettools.downloaders.HomusDatasetDownloader import HomusDatasetDownloader
from omrdatasettools.image_generators.HomusImageGenerator import HomusImageGenerator

dataset_downloader = HomusDatasetDownloader("./data/homus")
dataset_downloader.download_and_extract_dataset()


HomusImageGenerator.create_images(raw_data_directory="./data/homus",
                                  destination_directory="./data/homus/images",
                                  stroke_thicknesses=[3],
                                  canvas_width=96,
                                  canvas_height=192,
                                  staff_line_spacing=14,
                                  staff_line_vertical_offsets=[24])

#%% Printed MusicSymbols
from omrdatasettools.downloaders.PrintedMusicSymbolsDatasetDownloader import PrintedMusicSymbolsDatasetDownloader
dataset_downloader = PrintedMusicSymbolsDatasetDownloader("./data/printed")
dataset_downloader.download_and_extract_dataset()

#%% Audiveris
from omrdatasettools.downloaders.AudiverisOmrDatasetDownloader import AudiverisOmrDatasetDownloader
dataset_downloader = AudiverisOmrDatasetDownloader("./data/audiveris")
dataset_downloader.download_and_extract_dataset()

from omrdatasettools.image_generators.AudiverisOmrImageGenerator import AudiverisOmrImageGenerator, AudiverisOmrSymbol

imgen = AudiverisOmrImageGenerator()
imgen.extract_symbols(raw_data_directory='./data/audiveris', 
                                           destination_directory='./data/audiveris/images')

#%% Open OMR
from omrdatasettools.downloaders.OpenOmrDatasetDownloader import OpenOmrDatasetDownloader
 
dataset_downloader = OpenOmrDatasetDownloader("./data/openomr")
dataset_downloader.download_and_extract_dataset()

#%% Capitan
'''
from omrdatasettools.downloaders.CapitanDatasetDownloader import CapitanDatasetDownloader
dataset_downloader = CapitanDatasetDownloader("./data/capitan")
dataset_downloader.download_and_extract_dataset()
'''

#%% MUSCIMA
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader
 
dataset_downloader = MuscimaPlusPlusDatasetDownloader("./data/muscima")
dataset_downloader.download_and_extract_dataset()

from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator
imgen = MuscimaPlusPlusImageGenerator()
imgen.extract_and_render_all_symbol_masks(raw_data_directory='./data/muscima', destination_directory='./data/muscima/images')

#%% Robelo
from omrdatasettools.downloaders.RebeloMusicSymbolDataset1Downloader import RebeloMusicSymbolDataset1Downloader
dataset_downloader = RebeloMusicSymbolDataset1Downloader("./data/robelo")
dataset_downloader.download_and_extract_dataset()
