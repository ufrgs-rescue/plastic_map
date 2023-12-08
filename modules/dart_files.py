import numpy as np
import os
import operator
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import shutil


__name__ = "dart_files"


def extraction(path, export, file_names, s, color, status, polymer):
    """
    Receives source and destination paths, along with configuration information for simulations, reflected in the source directory organization structured according to the model: <source_folder/polymer/coverage_percentage/band/*simulation files*>. This method creates destination folders (if needed) and returns a list of export paths for files with identifiable names, following the model export/s/color/status/percent/polymer_band_percent.extension.

    :param path: str - source path of raw simulation
    :param export: str - destination path of raw simulation
    :param file_names: str - default filenames in simulation
    :param s: str - submersion of polymer in the format S<submersion in cm> (e.g., 'S2' means 2 cm of submersion)
    :param color: str - polymer color
    :param status: str - polymer status (Dry, Wet, or Submerged)
    :param polymer: str - polymer name

    :return paths: list of export paths for files with identifiable names, following the model export/s/color/status/percent/polymer_band_percent.extension
    """
    
    if path[len(path)-1] == "/":
        pass
    else:
        path = path+"/"
    
    if export[len(export)-1] == "/":
        pass
    else:
        export = export+"/"
        
    if os.path.exists(export):
        pass
    else:
        try:
            os.makedirs(export)
        except:
            print("Error trying to create directory ", export)
            
    paths = []
    
    percents = dict()
    percents_list = os.listdir(path)
    os.chdir(path)
    
    for percent in percents_list:
        os.chdir(percent)
        bands_list = os.listdir()
        bands = dict()
        for band in bands_list:
            os.chdir(band)
            bands[band] = os.listdir()

            for f in range(len(file_names)):
                ext = str(file_names[f]).split('.')[len(str(file_names[f]).split('.'))-1]
                paths.append([path+percent+'/'+band+'/BroadBand/'+band+'/'+band+'/BRF/ITERX/IMAGES_DART/'+file_names[f],
                         export+s+'/'+color+'/'+status+'/'+percent+'/',
                         polymer+'_'+band+'_'+percent+'.'+ext])

            os.chdir('../')
        os.chdir('../')

    percents[percent] = bands
    os.chdir('../')
  
    return paths



def get_directory_tree(path: str) -> dict:
    """
    Retrieves the directory tree structure from the specified source folder path.

    The method returns a dictionary representing the entire directory structure. Each path in
    the subfolders should adhere to the structure: source_folder/polymer/submersion_depth/color/status 
    (Dry, Wet, or Submerged)/cover_percent/dart .asc files (an individual file for each sensor band).

    :param path: str - The source path for dart .asc files, following the structure 
                  source_folder/polymer/submersion_depth/color/status (Dry, Wet, or Submerged)/
                  cover_percent/dart .asc files (an individual file for each sensor band).

    :return tree: dict - A dictionary containing the source path and the complete directory tree.
    """
    
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
  
    #return path, polymers
    return tree


def get_images(paths, resample_method, scaling_mode):
    """
    Retrieves the DART .asc files from the directory tree structure received using the resampling strategy defined.

    The method returns a set of Image objects representing the DART dataset. The directory tree should adhere to the structure: source_folder/polymer/submersion_depth/color/status (Dry, Wet, or Submerged)/cover_percent/arquivos .asc (an individual file for each sensor band). Os nomes dos arquivos precisam estar no padrão polimero_banda_percentualdecobertura.asc (ex: LDPE_Blue_100.asc). Os métodos de reamostragem podem ser interpolação bilinear ("bilinear"), interpolação cúbica ("cubic") ou vizinho mais próximo ("nearest", método padrão). scaling_mode pode ser "up" for upscale or "down" for downscale.  
            resampling = Resampling.cubic
        else: 
            resampling = Resampling.nearest
         

    :param path: str - The source path for dart .asc files, following the structure 
                  source_folder/polymer/submersion_depth/color/status (Dry, Wet, or Submerged)/
                  cover_percent/dart .asc files (an individual file for each sensor band).

    :return tree: dict - A dictionary containing the source path and the complete directory tree.
    """
    imagery = []
    for path in paths.values():
        polymers = path[1]
        for polymer in polymers.keys():
            submergences = polymers[polymer]
            for submergence in submergences.keys():
                colors = submergences[submergence]
                for color in colors.keys():
                    status = colors[color]
                    for stat in status.keys():
                        cover_percents = status[stat]
                        for percent in cover_percents:
                            directory = path+polymer+"/"+submergence+"/"+color+"/"+stat+"/"+percent+"/"
                            files_list = os.listdir(directory) 
                            if files_list:
                                bands = []
                                for dart_file in files_list:
                                    info = str(dart_file).split(sep="_")  
                                    i = len(info)-2
                                    if info[i] not in bands:
                                        bands.append(info[i])
                            current_image = Image(directory, files_list, polymer, submergence, color, stat, percent, bands, resample_method, scaling_mode)   
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
    column_names.append('Polymer')
    column_names.append('Cover_percent')
    column_names.append('Submergence')
    column_names.append('Color')
    column_names.append('Status')
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
            data.append(pixel.getPolymer())
            data.append(pixel.getCoverPercent())
            data.append(pixel.getSubmergence())
            data.append(pixel.getColor())
            data.append(pixel.getStatus())
            data.append(pixel.getLabel()) 
            dataset_dart.loc[i] = data
            i += 1
    return dataset_dart


class Image: 
    def __init__(self, path, file_names, polymer, submergence, color, status, percent, bands, resample_method, scaling_mode):
        print(path, file_names, polymer, submergence, color, status, percent, bands, resample_method, scaling_mode)
        self.setPath(path)
        self.setFileNames(file_names)
        self.setPolymer(polymer)
        self.setSubmergence(submergence)
        self.setColor(color)
        self.setStatus(status)
        self.setResampleMethod(resample_method)
        self.setScalingMode(scaling_mode)
        self.setBands(bands)
        self.setPlasticCoverPercent(percent)
        self.initLabelsMap()
        self.setPixels()

    def setPath(self, path):
        self.path = path
        
    def setFileNames(self, file_names):
        self.file_names = file_names
    
    def setPolymer(self, polymer):
        self.polymer = polymer
        
    def setSubmergence(self, submergence):
        self.submergence = submergence
        
    def setColor(self, color):
        self.color = color
        
    def setStatus(self, status):
        self.status = status
    
    def setResampleMethod(self, resample_method):
        self.resample_method = resample_method
        
    def setScalingMode(self, scaling_mode):
        if scaling_mode == "up" or scaling_mode == "down": 
            self.scaling_mode = scaling_mode
        else:
            print("Error: Scale mode must be equal to 'up' or 'down', otherwise it will not be possible to resample bands")
        
    def setPlasticCoverPercent(self, percent):
        self.plastic_cover_percent = percent

    def setBands(self, bands):
        self.bands = bands
        self.setBandsSizes()
            
    def setBandsSizes(self):
        self.bands_sizes = dict()
        for file_name in self.getFileNames():
            i = len(str(file_name).split(sep="_")) - 2
            current_band = str(file_name).split(sep="_")[i]
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
        self.setXSize(self.getBandsSizes()[best_resolution][0])
        self.setYSize(self.getBandsSizes()[best_resolution][1])
        
        if best_resolution != worst_resolution:
            if self.getScalingMode() == 'up':
                print("The ", self.getPath()," image bands will be resampled to the higher available spatial resolution " + str(self.getBandsSizes()[best_resolution]))
            elif self.getScalingMode() == 'down':
                print("The ", self.getPath()," image bands will be resampled to the lower available spatial resolution " + str(self.getBandsSizes()[worst_resolution]))
        else:
            print("All bands have the same resolution. There is no need to resample.")
            
        
        files = dict()
        for file_name in self.getFileNames():
            current_file = Band(self.getPath(), file_name)
            if self.getScalingMode() == "up":
                scale_factor_x = self.getBandsSizes()[best_resolution][0] / current_file.getResolution()[0]
                scale_factor_y = self.getBandsSizes()[best_resolution][1] / current_file.getResolution()[1]
                if scale_factor_x == scale_factor_y:
                    if scale_factor_x > 1:
                        files.update({current_file.getBandName(): current_file.resample(scale_factor_x, self.getResampleMethod())})
                    elif scale_factor_x == 1:
                        files.update({current_file.getBandName(): current_file.getData()})
                    else:
                        break
                else: 
                    print("Unable to resample because scale factor is different for x and y axes")
                    break
            elif self.getScalingMode() == "down":
                scale_factor_x = self.getBandsSizes()[worst_resolution][0] / current_file.getResolution()[0]
                scale_factor_y = self.getBandsSizes()[worst_resolution][1] / current_file.getResolution()[1]
                
                if scale_factor_x == scale_factor_y:
                    if scale_factor_x < 1:
                        files.update({current_file.getBandName(): current_file.resample(scale_factor_x, self.getResampleMethod())})
                    elif scale_factor_x == 1:
                        files.update({current_file.getBandName(): current_file.getData()})
                    else:
                        break
                        
                    self.setXSize(self.getBandsSizes()[worst_resolution][0])
                    self.setYSize(self.getBandsSizes()[worst_resolution][1])
                else: 
                    print("Unable to resample because scale factor is different for x and y axes")
                    break
                
        
        if len(files) == len(self.getFileNames()):
            self.pixels = self.mountMultiband(files)
        else:
            print("Error reading bands - unable to complete process")   
            

    def mountMultiband(self, files):
        pixels = []
        for n_line in range(self.getXSize()):
            for n_col in range(self.getYSize()):
                current_pixel = Pixel(n_line, n_col, self.getPath(), self.getPolymer(), self.getSubmergence(), self.getColor(), self.getStatus())
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
        for i in range(first_line, last_line + 1): #+1 is used because the range function does not include the informed number in the "stop" parameter
            for j in range(first_column, last_column + 1):
                area.append((i, j, label))
        self.setLabelsMap(area)
    
    def setGridLabel(self, first_line, last_line, line_step, first_column, last_column, column_step, label):
        grid_targets = []
        for i in range(first_line, last_line + 1, line_step): #+1 is used because the range function does not include the informed number in the "stop" parameter
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

    def getPolymer(self):
        return self.polymer
    
    def getSubmergence(self):
        return self.submergence
    
    def getColor(self):
        return self.color
    
    def getStatus(self):
        return self.status

    def getResampleMethod(self):
        return self.resample_method
    
    def getScalingMode(self):
        return self.scaling_mode
    
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
        i = len(str(self.getFileName()).split(sep="_")) - 2
        self.band_name = str(self.getFileName()).split(sep="_")[i]                       
        
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

    def resample(self, scale_factor, resample_method):
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
                int(dataset.height * scale_factor),
                int(dataset.width * scale_factor)
                ),
                resampling=resampling 
            )
        return data[0]

    
class Pixel:
    def __init__(self, line, col, image_name, polymer, submergence, color, status):
        self.setImageName(image_name)
        self.setLine(line)
        self.setColumn(col)
        self.setPolymer(polymer)
        self.setSubmergence(submergence)
        self.setColor(color)
        self.setStatus(status)
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
            
    def setPolymer(self, polymer):
        self.polymer = polymer
        
    def setSubmergence(self, submergence):
        self.submergence = submergence
        
    def setColor(self, color):
        self.color = color
        
    def setStatus(self, status):
        self.status = status

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
    
    def getPolymer(self):
        return self.polymer
    
    def getSubmergence(self):
        return self.submergence
    
    def getColor(self):
        return self.color
    
    def getStatus(self):
        return self.status

    def getLabel(self): 
        return self.label
    
    def getCoverPercent(self):
        return self.cover_percent
  
    def getReflectanceValues(self):
        return self.reflectance_values