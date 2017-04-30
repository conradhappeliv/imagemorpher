import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np


face1 = cv2.imread('data/face1.jpg')
face2 = cv2.imread('data/face2.jpg')
face1gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
face2gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)


# get from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
def find_points(img,shp_pred_file = 'data/shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shp_pred_file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    #find points of first face
    shape = predictor(gray, rects[0])

    points = []
    for pt in shape.parts():
        points.append((pt.x, pt.y))

    # Edge points:
    points.append((0, 0))
    points.append((img.shape[1]/2, 0))
    points.append((img.shape[1] - 1, 0))
    points.append((0, img.shape[0]/2))
    points.append((0, img.shape[0] - 1))
    points.append((img.shape[1] - 1, img.shape[0]/2))
    points.append((img.shape[1]/2, img.shape[0] - 1))
    points.append((img.shape[1] - 1, img.shape[0] - 1))
    return points


def weighted_points(points1, points2, alpha=0.5):
    #assume points1 and points2 same size, maybe should add size check
    points = []
    for pt1,pt2 in zip(points1,points2):
        x = (1-alpha)*pt1[0] + alpha*pt2[0]
        y = (1-alpha)*pt1[1] + alpha*pt2[1]
        points.append((x,y))
    return points


def find_triangles(shape, points):
    rect = (0, 0, shape[1], shape[0])
    subdiv = cv2.Subdiv2D(rect)
    [subdiv.insert(pt) for pt in points]
    return subdiv.getTriangleList()


def show_triangles(img, triangles):
    for t in triangles:
        tri1 = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        cv2.line(face1, tri1[0], tri1[1], (255, 255, 255))
        cv2.line(face1, tri1[1], tri1[2], (255, 255, 255))
        cv2.line(face1, tri1[2], tri1[0], (255, 255, 255))

    cv2.imshow("window", img)
    cv2.waitKey(0)

points1 = find_points(face1)
points2 = find_points(face2)
points_morphed = weighted_points(points1,points2)
# points_morphed = [((pt1[0]+pt2[0])/2, (pt1[1]+pt2[1])/2) for pt1, pt2 in zip(points1, points2)]

# plt.figure()
# plt.imshow(face1)
# for pt in points1:
#     plt.plot(pt[0], pt[1], 'o')
#
# plt.figure()
# plt.imshow(face2)
# for pt in points2:
#     plt.plot(pt[0], pt[1], 'o')
#
# plt.show()

#[4:] because 0,2,3 are insignificant, and 1 is basically copy of entire image
triangles1 = find_triangles(face1.shape, points1)[4:]
triangles2 = find_triangles(face2.shape, points2)[4:]
triangles_morphed = find_triangles(face2.shape, points_morphed)[4:]

# See: https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.py
newimg = np.zeros(face2gray.shape, np.uint8)
for tri1, tri2, tri_morph in zip(triangles1, triangles2, triangles_morphed):
    #3 arrays/edges, 2 elements(begin and end of edge) each
    tri1 = tri1.reshape((3,2))
    tri2 = tri2.reshape((3,2))
    tri_morph = tri_morph.reshape((3,2))
    #tri2 = np.array([(tri2[0], tri2[1]), (tri2[2], tri2[3]), (tri2[4], tri2[5])])
    #triangles matrices are numpy float32 type

    #rect object from cv2
    bb1 = cv2.boundingRect(tri1)
    bb2 = cv2.boundingRect(tri2)
    bb_morph = cv2.boundingRect(tri_morph)
    #regarding rect object (7.a.) http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
    #rect = [topleftx, toplefty, width, height]

    #find offset from each vertex of the triangle to the top right corner of respective bounding box
    #by default np subtract gives float64, also probably same speed as subtraction operator (also float64 by default)
    tri1_offset = np.subtract(tri1 , np.repeat([[bb_morph[0],bb_morph[1]]],3,axis=0)).astype(np.float32)
    tri2_offset = np.subtract(tri2 , np.repeat([[bb_morph[0],bb_morph[1]]],3,axis=0)).astype(np.float32)
    tri_morph_offset = np.subtract(tri_morph , np.repeat([[bb_morph[0],bb_morph[1]]],3,axis=0)).astype(np.float32)
    #tri2_offset = np.array([(tri2[0][0]-bb2[0], tri2[0][1]-bb2[1]), (tri2[1][0]-bb2[0], tri2[1][1]-bb2[1]), (tri2[2][0]-bb2[0], tri2[2][1]-bb2[1])], np.float32)

    #our mask is size bb_morph(height) x bb_morph(width)
    #thus our offsets must be taken with respect to that same bb_morph box
    #so when we affine traingles with respect to the same point, we can map them to the same box/mask
    trans1 = cv2.getAffineTransform(tri1_offset, tri_morph_offset)
    trans2 = cv2.getAffineTransform(tri2_offset, tri_morph_offset)

    mask = np.zeros((bb_morph[3], bb_morph[2]), np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_morph_offset), (1.0,1.0,1.0), 16,0)

    #bb  = [topleftx, toplefty, width, height]
    subimg1 = face1gray[bb1[1]:bb1[1]+bb1[3], bb1[0]:bb1[0]+bb1[2]]
    subimg2 = face2gray[bb2[1]:bb2[1]+bb2[3], bb2[0]:bb2[0]+bb2[2]]

    # SOMETHING IS WRONG WITH THIS SECTION (PROBABLY)
    warped1 = cv2.warpAffine(subimg1, trans1, (bb_morph[2], bb_morph[3]))
    warped2 = cv2.warpAffine(subimg2, trans2, (bb_morph[2], bb_morph[3]))
    plt.subplot(1,2,1)
    plt.imshow(warped1*mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(warped2*mask, cmap='gray')
    plt.show()

    newimg_sq = newimg[bb_morph[1]:bb_morph[1]+bb_morph[3], bb_morph[0]:bb_morph[0]+bb_morph[2]]
    newimg[bb_morph[1]:bb_morph[1]+bb_morph[3], bb_morph[0]:bb_morph[0]+bb_morph[2]] = newimg_sq + ((warped1*.5+warped2*.5)*mask)[:newimg_sq.shape[0], :newimg_sq.shape[1]]

plt.figure()
plt.imshow(newimg, cmap='gray')
plt.show()
