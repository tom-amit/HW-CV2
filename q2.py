import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import math as m
import imageio.v2 as imageio
import os


def loadData(filePath):
    arr = np.NAN
    with open(filePath, "r") as filestream:
        for line in filestream:
            currentline = [float(x) for x in line.split(",")]
            if arr is np.NAN:
                arr = np.array(currentline).reshape(1, len(currentline))
            else:
                arr = np.concatenate((arr, np.array(currentline).reshape(1, len(currentline))), axis=0)
    return arr


def drawPoints(im_list, pts_list):
    _, axs = plt.subplots(1, 2, figsize=(12, 12))
    for i, (im, ax, pts) in enumerate(zip(im_list, axs, pts_list)):
        ax.set_title("Image: " + str(i + 1))
        ax.scatter(pts[:, 0], pts[:, 1], marker="x", color="red", s=50)
        for j in range(pts.shape[0]):
            ax.plot(pts[j:, 0], pts[j:, 1], color=color[j], linewidth=1)
        ax.imshow(im)
    plt.show()


def triangulate(proj_list, pts_list):
    def DLT(P1, P2, point1, point2):
        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        print('Triangulated point: ')
        print(Vh[3, 0:3] / Vh[3, 3])
        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = []
    p1, p2 = pts_list
    projection1, projection2 = proj_list
    for uv1, uv2 in zip(p1, p2):
        _p3d = DLT(projection1, projection2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    return p3ds - np.mean(p3ds, axis=0)


def visualize_cloud_xy(cloud, display=True, save=False, saveIndex=0, title="Plot"):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([-4, 3])
    ax.set_ylim([-4, 4])
    ax.set_title(title)
    for j in range(cloud.shape[0]):
        ax.plot(cloud[j:, 0], cloud[j:, 1], color=color[j], linewidth=1)
    if save:
        plt.savefig("images/" + str(saveIndex) + '.png')
    if display:
        plt.show()
    plt.close()


def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, m.cos(theta), -m.sin(theta)],
                     [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.array([[m.cos(theta), 0, m.sin(theta)],
                     [0, 1, 0],
                     [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.array([[m.cos(theta), -m.sin(theta), 0],
                     [m.sin(theta), m.cos(theta), 0],
                     [0, 0, 1]])


def build_gif(path):
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in sorted(os.listdir("images/"), key=lambda x: int(os.path.splitext(x)[0])):
            image = imageio.imread(path + "/" + filename)
            writer.append_data(image)


if __name__ == "__main__":
    im1 = cv.imread("Q2/house_1.png")
    im2 = cv.imread("Q2/house_2.png")
    proj1, proj2 = loadData("Q2/cameraMatrix1.txt"), loadData("Q2/cameraMatrix2.txt")
    pts1, pts2 = loadData("Q2/matchedPoints1.txt"), loadData("Q2/matchedPoints2.txt")
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(pts1.shape[0])]

    drawPoints([im1, im2], [pts1, pts2])
    c = triangulate([proj1, proj2], [pts1, pts2])
    visualize_cloud_xy(c, title="Triangulation")

    funcs = [Rx, Ry, Rz]
    R_random = np.linalg.multi_dot([R(random.random() * 360) for R in funcs])
    c = np.dot(R_random, c.T).T

    visualize_cloud_xy(c, title="Random Rotation")
    i = 0
    iterations = 36
    angle = 2 * np.pi / iterations
    for rotation in range(iterations):
        c = np.dot(Rx(angle), c.T).T
        visualize_cloud_xy(c, display=False, save=True, saveIndex=i)
        i += 1
    for rotation in range(iterations):
        c = np.dot(Ry(angle), c.T).T
        visualize_cloud_xy(c, display=False, save=True, saveIndex=i)
        i += 1
    build_gif("images/")
