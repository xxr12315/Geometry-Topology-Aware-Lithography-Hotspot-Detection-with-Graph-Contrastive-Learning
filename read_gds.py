import sys
import matplotlib.pyplot as plt
import numpy as np
import gdstk
import os
from matplotlib.patches import Polygon
import json
import pickle

from torch.distributions import OneHotCategoricalStraightThrough
import os

# 遍历文件夹
def traversal_folder(folder_path):
    file_pa = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        file_pa.append(file_path)
    return file_pa

file_load = "D:\\Users\\WINDOWS\\PycharmProjects\\TagHot\\"


def image(corner, polygons):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 第一部分：小图形
    corner_closed = np.vstack([corner, corner[0]])
    ax.plot(*corner.T, color='red', linestyle='--', linewidth=2, label='Bounding Box')

    for poly in polygons:
        patch = Polygon(poly, closed=True, facecolor='black', edgecolor='black', linewidth=1)
        ax.add_patch(patch)

    ax.set_title("Hotspot Geometry with Bounding Box")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
    plt.clf()  # 清空当前图像（Figure），但窗口仍保留
    plt.close()




def get_all_cells(gds_path):
    # 读取 GDS/OAS 文件
    if gds_path.endswith(".gds"):
        print(gds_path)
        lib = gdstk.read_gds(gds_path)
    elif gds_path.endswith(".oas"):
        lib = gdstk.read_oas(gds_path)
    else:
        raise ValueError("仅支持 .gds 或 .oas 文件")

    # 获取所有 cell：lib.cells 是一个 dict

    all_cells = lib.cells  # dict: {name (str): Cell}

    return all_cells



def read_2012_test_dataset(data2012,num2012,dataset_type):
    data_set2012 = get_all_cells(data2012)
    poly = data_set2012[0].polygons
    hot_data = {}
    nonhot_data= {}
    hsnumber = 0
    nhsnumber = 0

    for polygon in poly:
        if polygon.layer == 21 or polygon.layer == 22:
            hsnumber += 1
            new_cor = []
            for cell in poly:
                if cell.layer == 10:
                    for item in gdstk.boolean(polygon,cell,"and"):
                        new_cor.append(item.points.tolist())
            hot_data[str(polygon.points)] = new_cor

        if polygon.layer == 23:
            nhsnumber += 1
            new_cor = []
            for cell in poly:
                if cell.layer == 10:
                    for item in gdstk.boolean(polygon, cell, "and"):
                        new_cor.append(item.points.tolist())
            nonhot_data[str(polygon.points)] = new_cor


    with open("ICCAD2012\\"+str(num2012)+"\\ICCAD2012_dataset_hospot_"+str(num2012)+"_"+dataset_type+".json", "w", encoding="utf-8") as f:
        json.dump(hot_data, f, ensure_ascii=False, indent=4)
    with open("ICCAD2012\\"+str(num2012)+"\\ICCAD2012_dataset_nonhospot_" + str(num2012)+"_"+dataset_type + ".json", "w", encoding="utf-8") as f:
        json.dump(nonhot_data, f, ensure_ascii=False, indent=4)


    print(dataset_type+"中共有" + str(len(hot_data)) + "个热点模式")
    print(dataset_type+"中共有" + str(len(nonhot_data)) + "个非热点模式")

def process_gds_2012():
    for i in range(5):
        read_2012_test_dataset("ICCAD2012\\" + str(i + 1) + "\\test.gds", i + 1, "test")
        read_2012_test_dataset("ICCAD2012\\"+str(i + 1)+"\\train.gds", i+1, "train")

def process_gds_2019():
    train_data = "ICCAD2019\\training_dataset\\training_dataset.oas"
    dateset2019_train = get_all_cells(train_data)
    hot_data = {}
    nonhot_data= {}
    hsnumber = 0
    nhsnumber = 0
    for cell in dateset2019_train:
        for poly in cell.polygons:
            if poly.layer == 21 or poly.layer == 22:
                hsnumber += 1
                new_cor = []
                for polygon in cell.polygons:
                    if polygon.layer == 10:
                        for item in gdstk.boolean(polygon, poly, "and"):
                            new_cor.append(item.points.tolist())
                hot_data[str(poly.points)] = new_cor

            if poly.layer == 23:
                nhsnumber += 1
                new_cor = []
                for polygon in cell.polygons:
                    if polygon.layer == 10:
                        for item in gdstk.boolean(polygon, poly, "and"):
                            new_cor.append(item.points.tolist())
                nonhot_data[str(poly.points)] = new_cor

    with open("ICCAD2019\\ICCAD2019_train_dataset_hospot.json", "w", encoding="utf-8") as f:
        json.dump(hot_data, f, ensure_ascii=False, indent=4)
    with open("ICCAD2019\\ICCAD2019_train_dataset_nonhospot.json", "w", encoding="utf-8") as f:
        json.dump(nonhot_data, f, ensure_ascii=False, indent=4)

    print("训练集中共有" + str(len(hot_data)) + "个热点模式")
    print("训练集中共有" + str(len(nonhot_data)) + "个非热点模式")


    test_data = "ICCAD2019\\testing_dataset_1\\testing_dataset_1.oas"
    dateset2019_test = get_all_cells(test_data)
    hot_data = {}
    nonhot_data= {}
    hsnumber = 0
    nhsnumber = 0
    for cell in dateset2019_test:
        for poly in cell.polygons:
            if poly.layer == 21 or poly.layer == 22:
                hsnumber += 1
                new_cor = []
                for polygon in cell.polygons:
                    if polygon.layer == 10:
                        for item in gdstk.boolean(polygon, poly, "and"):
                            new_cor.append(item.points.tolist())
                hot_data[str(poly.points)] = new_cor

            if poly.layer == 23:
                nhsnumber += 1
                new_cor = []
                for polygon in cell.polygons:
                    if polygon.layer == 10:
                        for item in gdstk.boolean(polygon, poly, "and"):
                            new_cor.append(item.points.tolist())
                nonhot_data[str(poly.points)] = new_cor

    with open("ICCAD2019\\ICCAD2019_test_dataset_hospot.json", "w", encoding="utf-8") as f:
        json.dump(hot_data, f, ensure_ascii=False, indent=4)
    with open("ICCAD2019\\ICCAD2019_test_dataset_nonhospot.json", "w", encoding="utf-8") as f:
        json.dump(nonhot_data, f, ensure_ascii=False, indent=4)

    print("测试集中共有" + str(len(hot_data)) + "个热点模式")
    print("测试集中共有" + str(len(nonhot_data)) + "个非热点模式")

    test_data2 = "ICCAD2019\\testing_dataset_2"
    hot_data = {}
    nonhot_data = {}
    hsnumber = 0
    nhsnumber = 0
    for i in traversal_folder(test_data2):
        dateset2019_test = get_all_cells(i)
        for cell in dateset2019_test:
            for poly in cell.polygons:
                if poly.layer == 21 or poly.layer == 22:
                    hsnumber += 1
                    new_cor = []
                    for polygon in cell.polygons:
                        if polygon.layer == 10:
                            for item in gdstk.boolean(polygon, poly, "and"):
                                new_cor.append(item.points.tolist())
                    hot_data[str(poly.points)] = new_cor

                if poly.layer == 23:
                    nhsnumber += 1
                    new_cor = []
                    for polygon in cell.polygons:
                        if polygon.layer == 10:
                            for item in gdstk.boolean(polygon, poly, "and"):
                                new_cor.append(item.points.tolist())
                    nonhot_data[str(poly.points)] = new_cor

    with open("ICCAD2019\\ICCAD2019_2_test_dataset_hospot.json", "w", encoding="utf-8") as f:
        json.dump(hot_data, f, ensure_ascii=False, indent=4)
    with open("ICCAD2019\\ICCAD2019_2_test_dataset_nonhospot.json", "w", encoding="utf-8") as f:
        json.dump(nonhot_data, f, ensure_ascii=False, indent=4)

    print("训练集2中共有" + str(len(hot_data)) + "个热点模式")
    print("训练集2中共有" + str(len(nonhot_data)) + "个非热点模式")


if __name__ == "__main__":

    process_gds_2019()

    #process_gds_2012()

    #ceshi()


