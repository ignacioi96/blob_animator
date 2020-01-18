The first fruit that has been born from image explorations, a project that
started long ago to try and see what could be done with images. This code was
the main ingredient for an animation project for a Gender, Race and Social
Justice course. The project tried to speak out on the illusory nature of our
perceptions, preconceptions, assumptions, judgementalities and the likes,
especially those related to Gender, Race and Social Justice.

The process is simple. A directory holding images and a storing directory are
provided in the code, in addition to a single image of the object that must be
found and "animated".

![Fig 1. The object that we want to be "animated".](figures/object_img.png)

The images are analyzed with the intent of finding the object using SIFT
features.

![Fig 2. One of the images provided.](figures/original_img.png)
![Fig 3. An example of SIFT feature matching.](figures/sift_example.png)

After finding the desired object in each of the singular images, it will search
for blobs of varying sizes and shapes using a Laplassian of Gaussian method.

![Fig 4. Different blobs identified and colored in.](figures/all_blobs.png)

Each of this blobs is abstracted to stand on its own, i.e. in a separate image,
everything that is not the blob is made a single color.

![Fig 5. Examples of blobs.](figures/blob_1.png) ![](figures/blob_2.png) ![](blob_3.png)

Then, each of these "blob images" are transformed in various ways e.g. rotated,
shrunk, etc. and the image is stiched back together with the modified object
in it.

![Fig 6. The original image with the modified object in it.](figures/mod_img.png)

Option to make gifs, and videos are pausible, but at the moment the video option
does not function as expected.

The video I created for the class is available upon request.

![Fig 7. A jittery example gif.](figures/example.gif)
