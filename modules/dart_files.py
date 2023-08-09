import numpy as np
import os
import operator
import pandas as pd
import rasterio
from rasterio.enums import Resampling


__name__ = "dart_files"


def open_folders(path):
    
    if path[len(path)-1] == "/":
        pass
    else:
        path = path+"/"

    polymers = dict()
    polymers_list = os.listdir(path)
    os.chdir(path)
    
    for polymer in polymers_list:
        os.chdir(polymer)
        submergences_list = os.listdir()
        submergences = dict()
        for submergence in submergences_list:
            os.chdir(submergence)
            colors_list = os.listdir()
            colors = dict()
            for color in colors_list:
                os.chdir(color)
                status_list = os.listdir()
                status = dict()
                for stat in status_list:
                    os.chdir(stat)
                    status[stat] = os.listdir()
                    os.chdir('../')
                colors[color] = status
                os.chdir('../')
            submergences[submergence] = colors
            os.chdir('../')
        polymers[polymer] = submergences
        os.chdir('../')
  
    return path, polymers


def get_images(path, polymers, cover_percent, resample_method):
    imagery = []
    for polymer in polymers:
        for cover in cover_percent:
            if cover == polymer:
                percents = cover_percent.get(cover)
                for percent in percents:
                    #directory = path+polymer+"/"+percent+"/"
                    directory = polymer+"/"+percent+"/"
                    #print(os.listdir())
                    files_list = os.listdir(directory) 
                    if files_list:
                        bands = []
                        for dart_file in files_list:
                            info = str(dart_file).split(sep="_")                         
                            if info[1] not in bands:
                                bands.append(info[1]) 
                        percent = str(info[2]).split(sep=".")[0] 
                        #info[2] corresponde ao percentual de cobertura informado no nome do último arquivo - pressupõe-se que todos dentro da pasta tem o mesmo percentual de cobertura (pode ser necessário verificar)
                        current_image = Image(directory, files_list, percent, bands, resample_method)   #AQUI - DAR OPCAO DE REAMOSTRAGEM
                        imagery.append(current_image)               
    return imagery


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
    column_names.append('Cover_percent')
    column_names.append('Label')

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
            data.append(pixel.getCoverPercent())
            data.append(pixel.getLabel()) 
            dataset_dart.loc[i] = data
            i += 1
    return dataset_dart


class Image:
    def __init__(self, path, file_names, percent, bands, resample_method):
        self.setPath(path)
        self.setFileNames(file_names)
        self.setResampleMethod(resample_method)
        self.setBands(bands)
        self.setPlasticCoverPercent(percent)
        self.initLabelsMap()
        self.setPixels()

    def setPath(self, path):
        self.path = path
        
    def setFileNames(self, file_names):
        self.file_names = file_names
    
    def setResampleMethod(self, resample_method):
        self.resample_method = resample_method
        
    def setPlasticCoverPercent(self, percent):
        self.plastic_cover_percent = percent

    def setBands(self, bands):
        self.bands = bands
        self.setBandsSizes()
            
    def setBandsSizes(self):
        self.bands_sizes = dict()
        for file_name in self.getFileNames():
            current_band = str(file_name).split(sep="_")[1]
            data = open(self.getPath()+"/"+file_name)
            data = [x.split() for x in data]
            data = data[6:]
            self.bands_sizes.update({current_band: (len(data), len(data[0]))})
            
    def setXSize(self, x_size):
        self.x_size = x_size
    
    def setYSize(self, y_size):
        self.y_size = y_size
        
    def setPixels(self):
        best_resolution = max(self.getBandsSizes(), key = self.getBandsSizes().get)
        worst_resolution = min(self.getBandsSizes(), key = self.getBandsSizes().get)
        #É necessário garantir que as imagens sejam "quadradas" (120x120, 30x30, 60x60, etc) - caso contrário cálculo não funciona
        self.setXSize(self.getBandsSizes()[best_resolution][0])
        self.setYSize(self.getBandsSizes()[best_resolution][1])
        
        if best_resolution != worst_resolution:
            print("The ", self.getPath()," image bands will be resampled to the best available spatial resolution " + str(self.getBandsSizes()[best_resolution]))
        
        files = dict()
        for file_name in self.getFileNames():
            current_file = Band(self.getPath(), file_name)
            upscale_factor = int(self.getBandsSizes()[best_resolution][0] / current_file.getResolution()[0])
            if upscale_factor > 1:
                files.update({current_file.getBandName(): current_file.resample(upscale_factor, self.getResampleMethod())})
            elif upscale_factor == 1:
                files.update({current_file.getBandName(): current_file.getData()})
            else:
                break        

        if len(files) == len(self.getFileNames()):
            self.pixels = self.mountMultiband(files)
        else:
            print("Erro na leitura das bandas - não é possível concluir o processo")    

    def mountMultiband(self, files):
        pixels = []
        for n_line in range(self.getXSize()):
            for n_col in range(self.getYSize()):
                current_pixel = Pixel(n_line, n_col, self.getPath())
                for key in files:
                    current_pixel.setReflectanceValues(key, files[key][n_line][n_col])                   
                pixels.append(current_pixel)
        return pixels
    
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
        self.setLabelsMap([line, column, label])

    def getPath(self):
        return self.path

    def getBands(self):
        return self.bands

    def getFileNames(self):
        return self.file_names

    def getResampleMethod(self):
        return self.resample_method
    
    def getPlasticCoverPercent(self):
        return self.plastic_cover_percent
    
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
                  

class Band:
    def __init__(self, path, file_name):
        self.setPath(path)
        self.setFileName(file_name)
        self.setBandName()
        self.setData()
        self.setResolution()
    
    def setPath(self, path):
        self.path = path
    
    def setFileName(self, file_name):
        self.file_name = file_name
    
    def setBandName(self):
        self.band_name = str(self.getFileName()).split(sep="_")[1]                       
        
    def setData(self):
        data = open(self.getPath()+"/"+self.getFileName())
        data = [x.split() for x in data]
        self.data = data[6:]
    
    def setResolution(self):
        self.resolution = (len(self.getData()), len(self.getData()[0]))
    
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
        else: 
            resampling = Resampling.nearest
            
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

    def setCoverPercent(self, target_label, percent):
        if self.getLabel() == target_label:
            self.cover_percent = percent
        else:
            self.cover_percent = 0

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
  
    def getReflectanceValues(self):
        return self.reflectance_values