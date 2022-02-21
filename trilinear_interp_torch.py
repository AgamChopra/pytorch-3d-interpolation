from matplotlib import pyplot as plt
from torch import tensor, zeros, rand, where, int32, long


def triliniar(Bmap, image):  # Bmap(3,x,y,z), image(C,xi,yi,zi)

    warped_image = zeros((image.shape[0], Bmap.shape[1], Bmap.shape[2], Bmap.shape[3])).to(device=image.device)  # (C,x,y,z)

    for i in range(Bmap.shape[1]):
        for j in range(Bmap.shape[2]):
            for k in range(Bmap.shape[3]):
                x, y, z = int(Bmap[0, i, j, k]), int(
                    Bmap[1, i, j, k]), int(Bmap[2, i, j, k])

                if x >= 0 and y >= 0 and z >= 0 and x < image.shape[1] and y < image.shape[2] and z < image.shape[3]:
                    if x < image.shape[1]-1 and y < image.shape[2]-1 and z < image.shape[3]-1:
                        a = Bmap[0, i, j, k] - x
                        b = Bmap[1, i, j, k] - y
                        c = Bmap[2, i, j, k] - z
                        warped_image[:, i, j, k] = (1-a)*(1-b)*(1-c)*image[:, x, y, z] + (1-a)*(1-b)*(c)*image[:, x, y, z+1] + (1-a)*(b)*(1-c)*image[:, x, y+1, z] + (1-a)*(b)*(c)*image[:, x, y+1, z+1] + (a)*(1-b)*(1-c) * image[:, x+1, y, z] + (a)*(1-b)*(c)*image[:, x+1, y, z+1] + (a)*(b)*(1-c)*image[:, x+1, y+1, z] + (a)*(b)*(c)*image[:, x+1, y+1, z+1]
                    else:
                        x, y, z = i if x < image.shape[1]-1 else i-1, j if y < image.shape[2] - \
                            1 else j-1, k if z < image.shape[3]-1 else k-1
                        warped_image[:, i, j, k] = warped_image[:, x, y, z]

    return warped_image.to(int32)


def triliniar_vectorized(Bmap, image):  # Bmap(3,x,y,z), image(C,xi,yi,zi)

    warped_image = zeros((image.shape[0], Bmap.shape[1], Bmap.shape[2], Bmap.shape[3])).to(device=image.device)  # (C,x,y,z)

    x, y, z = Bmap[0].to(int32), Bmap[1].to(int32), Bmap[2].to(int32)

    a = Bmap[0] - x
    b = Bmap[1] - y
    c = Bmap[2] - z

    x, y, z = x.to(dtype=long), y.to(dtype=long), z.to(dtype=long)

    warped_image = (1-a)*(1-b)*(1-c)*image[:, x, y, z] + (1-a)*(1-b)*(c)*image[:, x, y, z+1] + (1-a)*(b)*(1-c)*image[:, x, y+1, z] + (1-a)*(b)*(c)*image[:, x, y+1, z+1] + (a)*(1-b)*(1-c)*image[:, x+1, y, z] + (a)*(1-b)*(c)*image[:, x+1, y, z+1] + (a)*(b)*(1-c)*image[:, x+1, y+1, z] + (a)*(b)*(c)*image[:, x+1, y+1, z+1]

    return warped_image.to(int32)


def main():
    
    img = where(rand((3, 10, 10, 5)) > 0.90, 255, 0).to(dtype=int32)  # (C,x,y,z)
    
    for i in range(img.shape[-1]):
        plt.imshow(img[:, :, :, i].T.cpu().detach().numpy())
        plt.show()
    
    Bmap = zeros((3, 50, 50, 10))
    
    for i in range(Bmap.shape[1]):
        for j in range(Bmap.shape[2]):
            for k in range(Bmap.shape[3]):
                Bmap[:, i, j, k] = tensor([(img.shape[1]-2)*i/Bmap.shape[1], (img.shape[2]-2)* j/Bmap.shape[2], (img.shape[3]-2)*k/Bmap.shape[3]])
                
    img_ = triliniar_vectorized(Bmap, img)
    
    for i in range(img_.shape[-1]):
        plt.imshow(img_[:, :, :, i].T.cpu().detach().numpy())
        plt.show()

        
if __name__ == '__main__':
    main()
