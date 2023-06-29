import gdal
from matplotlib import pyplot
import os
import pandas as pd
import rasterio

__name__ = "tiff_files"

def open_folders(path):
    
    if path[len(path)-1] == "/":
        pass
    else:
        path = path+"/"
    
    sources = os.listdir(path)
    dates = dict()
    os.chdir(path)

    for source in sources:
        if os.path.isdir(source):
            os.chdir(source)
            days = os.listdir()
            dates[source] = days
            os.chdir('../')
        
    return path, sources, dates


def get_images(path, sources, dates):
    imagery = []
    for source in sources:
        days = dates.get(source)
        for day in days:
            files = os.listdir(path+source+"/"+day)
            bands = []
            files_list = []
            for file in files:
                if Image.isTiffImage(file):
                    files_list.append(file)
                    band = str(file).split('.')[0]
                    band = band.split('_')[len(band.split('_'))-1]
                    bands.append(band)
            bands = list(pd.Series(bands).unique())
            current_image = Image(path, source, day, files_list, bands)   #AQUI - DAR OPCAO DE REAMOSTRAGEM
            imagery.append(current_image)               
    return imagery
    #if path[len(path)-1] == "/":
    #    pass
    #else:
    #    path = path+"/"
    
    #os.chdir(path)
    #data = []
    #
    #for folder in datasets.keys():
    #    os.chdir(folder)
    #    for file in os.listdir():
    #            if Image.isTiffImage(file):
    #                file_name = file[0:len(file)-4]
    #                data.append(str(file_name).split('_'))
                    
    #    os.chdir('../')
    
    #os.chdir('../') #talvez quebrar em dois métodos para separar a navegação nos diretórios
    #os.chdir('../')
    
    #data = pd.DataFrame(data, columns=['Source', 'Date', 'Product/ROI', 'Band'])
    
    #imagery = []
    
    #for date in list(data['Date'].unique()):
    #    os.chdir(path)
    #    for source in list(data.loc[data['Date']==date]['Source'].unique()):
    #        os.chdir(source)
    #        files_list = os.listdir()
    #        bands = []
    #        for tiff_file in files_list:
    #            info = tiff_file[0:len(tiff_file)-4]
    #            info = str(info).split(sep="_")                         
    #            if info[len(info)-1] not in bands:
    #                bands.append(info[len(info)-1]) 
    #        current_image = Image(path, files_list, bands, resample_method)   
    #        imagery.append(current_image)
    #        os.chdir('../')
    #    os.chdir('../')
    #    os.chdir('../')
    
    #return data, imagery

def build_dataset(image_collection):
    bands = []

    for image in image_collection:
        for band in image.getBands():
            if band not in bands:
                bands.append(band)

    column_names = []
    column_names.append('Path')
    column_names.append('Line')
    column_names.append('Column') 
    for band in bands:
        column_names.append(band)
    column_names.append('Label')
    column_names.append('Cover_percent')
    column_names.append('Polymer')
    
    dataset_dart = pd.DataFrame([], columns=column_names)
    
    for image in image_collection:
        i = len(dataset_dart)
        for pixel in image.getPixels(): 
            data = []
            data.append(pixel.getImageName())
            data.append(pixel.getLine())
            data.append(pixel.getColumn())
            for band in bands:
                if band in pixel.getReflectanceValues():
                    data.append(round(float(pixel.getReflectanceValues()[band]), 4))
                else:
                    data.append("Ausente")
            data.append(pixel.getLabel())
            data.append(pixel.getCoverPercent())
            data.append(pixel.getPolymer())
            dataset_dart.loc[i] = data
            i += 1
    return dataset_dart

        
class Image():
    def __init__(self, path, source, date, file_names, bands):
        self.setPath(path)
        self.setFileNames(file_names)
        self.setSource(source)
        self.setDate(date)
        self.setBands(bands)
        self.initLabelsMap()
        #self.setResampleMethod(resample_method)#Tem que reamostrar aqui
        self.setPixels()
        
    def setPath(self, path):
        self.path = path
        
    def setFileNames(self, file_names):
        self.file_name = file_names
        
    def setDate(self, date):
        self.date = date
        
    def setSource(self, source):
        self.source = source
    
    def setBands(self, bands):
        self.bands = bands
        self.setBandsSizes()
        
    def setBandsSizes(self):
        self.bands_sizes = dict()
        for file_name in self.getFileNames():
            current_band = str(file_name).split(sep="_")[3]
            current_band = current_band.split(sep=".")[0]
            data = rasterio.open(self.getPath()+self.getSource()+"/"+self.getDate()+"/"+file_name)
            self.bands_sizes.update({current_band: data.shape})
            
    def setXSize(self, x_size):
        self.x_size = x_size
    
    def setYSize(self, y_size):
        self.y_size = y_size
    
    def setResampleMethod(self, resample_method):
        self.resample_method = resample_method
    
    def setPixels(self):
        best_resolution = max(self.getBandsSizes(), key = self.getBandsSizes().get)
        worst_resolution = min(self.getBandsSizes(), key = self.getBandsSizes().get)
        #É necessário garantir que as imagens sejam "quadradas" (120x120, 30x30, 60x60, etc) - caso contrário cálculo não funciona
        self.setXSize(self.getBandsSizes()[best_resolution][0])
        self.setYSize(self.getBandsSizes()[best_resolution][1])
        
        if best_resolution != worst_resolution:
            print("As bandas da imagem ", self.getPath(), "precisam ser reamostradas antes do processamento")
        else:
            files = dict()
            for file_name in self.getFileNames():
                current_file = Band(self.getPath()+self.getSource()+"/"+self.getDate()+"/", file_name)
                files.update({current_file.getBandName(): current_file.getData()})
                
            self.pixels = self.mountMultiband(files)

    def mountMultiband(self, files):
        pixels = []
        for n_line in range(self.getXSize()):
            for n_col in range(self.getYSize()):
                current_pixel = Pixel(n_line, n_col, self.getDate())
                for key in files:
                    current_pixel.setReflectanceValues(key, files[key][n_line][n_col])                   
                pixels.append(current_pixel)
        return pixels
    
    @staticmethod
    def isTiffImage(file_name):
        if str(file_name).split('.')[len(str(file_name).split('.'))-1] == "tif":
            return True
        else:
            return False
        
    def initLabelsMap(self):
        self.labels_map = pd.DataFrame([], columns=['Line', 'Column', 'Label'])
        
    def setLabelsMap(self, data):
        for item in data:
            index = self.labels_map.index[(self.labels_map['Line'] == item[0]) & (self.labels_map['Column'] == item[1])].tolist()
            if index:
                for i in index:
                    self.labels_map.at[i, 'Label'] = item[2]
            else:
                j = len(self.getLabelsMap())
                self.labels_map.loc[j] = item
                
    def setAreaLabel(self, first_line, last_line, first_column, last_column, label):
        area = []
        for i in range(first_line, last_line + 1): #+1 é pq funcao range nao inclui o numero informado no parametro stop
            for j in range(first_column, last_column + 1):
                area.append((i, j, label))
        self.setLabelsMap(area)
    
    def setGridLabel(self, first_line, last_line, line_step, first_column, last_column, column_step, label):
        grid_targets = []
        for i in range(first_line, last_line + 1, line_step): #+1 é pq funcao range nao inclui o numero informado no parametro stop
            for j in range(first_column, last_column + 1, column_step):
                grid_targets.append((i, j, label))
        self.setLabelsMap(grid_targets)
    
    def setPixelLabel(self, line, column, label):
        self.setLabelsMap([[line, column, label]])
        
    def getPath(self):
        return self.path

    def getBands(self):
        return self.bands
    
    def getDate(self):
        return self.date
    
    def getSource(self):
        return self.source

    def getFileNames(self):
        return self.file_name
    
    def getBandsSizes(self):
        return self.bands_sizes
    
    def getXSize(self):
        return self.x_size
    
    def getYSize(self):
        return self.y_size
    
    def getPixels(self):
        return self.pixels
    
    def getLabelsMap(self):
        return self.labels_map

    def getResampleMethod(self):
        return self.resample_method
    
    
class Band:
    def __init__(self, path, file_name):
        self.setPath(path)
        self.setFileName(file_name)
        self.setBandName()
        self.setData()
        self.setResolution()
    
    def setPath(self, path):
        self.path = path
        #print("Path: ", path)
    
    def setFileName(self, file_name):
        self.file_name = file_name
        #print("File name: ", file_name)
    
    def setBandName(self):
        self.band_name = str(self.getFileName()).split(sep=".")[0]         
        self.band_name = str(self.band_name).split(sep="_")[len(str(self.band_name).split(sep="_"))-1]   
        #print("Band name: ", self.band_name)
        
    def setData(self):
        data = rasterio.open(self.getPath()+self.getFileName())
        self.data = data.read(1)
        #print("Data: ", data)
        data.close()
    
    def setResolution(self):
        self.resolution = (self.getData().shape[0], self.getData().shape[1])
        #print("Resolution: ", self.resolution)
        #print("******************************")
    
    def getPath(self):
        return self.path
    
    def getFileName(self):
        return self.file_name
    
    def getBandName(self):
        return self.band_name 
                             
    def getData(self):
        return self.data
    
    def getResolution(self):
        return self.resolution

    def resample(self, upscale_factor, resample_method):
        if resample_method == "bilinear":
            resampling = Resampling.bilinear
        elif resample_method == "cubic":
            resampling = Resampling.cubic
        elif resample_method == "rms":
            resampling = Resampling.rms
            
        with rasterio.open(self.getPath()+"/"+self.getFileName()) as dataset: 
            data = dataset.read(
                out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
                ),
                resampling=resampling 
            )
        return data[0]
    
    
    
class Pixel:
    def __init__(self, line, col, image_name):
        self.setImageName(image_name)
        self.setLine(line)
        self.setColumn(col)
        self.initReflectanceValues()
  
    def setImageName(self, image_name):
        self.image_name = image_name
        
    def setLine(self, line):
        self.line = line
  
    def setColumn(self, column):
        self.column = column
  
    def setLabel(self, image_labelsmap):
        index = image_labelsmap.index[(image_labelsmap['Line'] == self.getLine()) & (image_labelsmap['Column'] == self.getColumn())].tolist()
        if len(index) > 1:
            print("Registro duplicado em ", self.getLine(), self.getColumn())
        elif len(index) == 1:
            self.label = image_labelsmap.loc[index[0]]['Label']
        else:
            self.label = "Undefined"

    def setCoverPercent(self, percent):
        self.cover_percent = percent
    
    def setPolymer(self, polymer):
        self.polymer = polymer

    def initReflectanceValues(self):
        self.reflectance_values = {}

    def setReflectanceValues(self, band, reflectance_values):
        self.reflectance_values.update({band: reflectance_values})
  
    def getImageName(self):
        return self.image_name
    
    def getLine(self):
        return self.line

    def getColumn(self):
        return self.column

    def getLabel(self): 
        return self.label
    
    def getCoverPercent(self):
        return self.cover_percent
    
    def getPolymer(self):
        return self.polymer
  
    def getReflectanceValues(self):
        return self.reflectance_values