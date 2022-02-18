from matplotlib import pyplot as plt
import torch
from torch import tensor, zeros, rand, where


def triliniar(Bmap, image):

    warped_image = zeros((Bmap.shape[0], Bmap.shape[1], Bmap.shape[2], image.shape[-1])).to(device=image.device)

    for i in range(Bmap.shape[0]):
        for j in range(Bmap.shape[1]):
            for k in range(Bmap.shape[2]):
                x, y, z = int(Bmap[i, j, k, 0]), int(Bmap[i, j, k, 1]), int(Bmap[i, j, k, 2])

                if x >= 0 and y >= 0 and z >= 0 and x < image.shape[0] and y < image.shape[1] and z < image.shape[2]:
                    if x < image.shape[0]-1 and y < image.shape[1]-1 and z < image.shape[2]-1:
                        a = Bmap[i, j, k, 0] - x
                        b = Bmap[i, j, k, 1] - y
                        c = Bmap[i, j, k, 2] - z
                        warped_image[i, j, k] = (1-a)*(1-b)*(1-c)*image[x, y, z] + (1-a)*(1-b)*(c)*image[x, y, z+1] + (1-a)*(b)*(1-c)*image[x, y+1, z] + (1-a)*(b)*(c)*image[x, y+1, z+1] + (a)*(1-b)*(1-c) * image[x+1, y, z] + (a)*(1-b)*(c)*image[x+1, y, z+1] + (a)*(b)*(1-c)*image[x+1, y+1, z] + (a)*(b)*(c)*image[x+1, y+1, z+1]
                    else:
                        x, y, z = i if x < image.shape[0]-1 else i-1, j if y < image.shape[1] - 1 else j-1, k if z < image.shape[2]-1 else k-1
                        warped_image[i, j, k] = warped_image[x, y, z]

    return warped_image.to(torch.int32)


img = where(rand((5, 5, 3, 3)) > 0.90, 255, 0).to(dtype=torch.int32) #(z,y,x,1,N)

for i in range(img.shape[2]):
    plt.imshow(img[:, :, i].cpu().detach().numpy())
    plt.show()

Bmap = zeros((50, 50, 6, 3)) 

for i in range(Bmap.shape[0]):
    for j in range(Bmap.shape[1]):
        for k in range(Bmap.shape[2]):
            Bmap[i, j, k] = tensor([(img.shape[0]-1)*i/Bmap.shape[0], (img.shape[1]-1)
                                   * j/Bmap.shape[1], (img.shape[2]-1)*k/Bmap.shape[2]])
'''
img_ = nearest(Bmap, img)

for i in range(img_.shape[2]):
    plt.imshow(img_[:,:,i])
    plt.show()
'''
img_ = triliniar(Bmap, img)

for i in range(img_.shape[2]):
    plt.imshow(img_[:, :, i].cpu().detach().numpy())
    plt.show()
print(img_)