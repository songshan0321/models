from os import listdir
from os.path import isfile, join, splitext
from random import seed, shuffle

class TxtGenerator:
    
    def __init__(self, image_dir, output_dir, trainval_ratio, seed_int):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.trainval_ratio = trainval_ratio
        self.filenames = []
        self.trainval_filenames = []
        self.test_filenames = []
        self.seed_int = seed_int

    def read_images(self):
        self.filenames = [f for f in listdir(self.image_dir) if isfile(join(self.image_dir, f))]

    def split(self):
        self.filenames.sort()
        seed(self.seed_int)
        shuffle(self.filenames)
        split = int(self.trainval_ratio * len(self.filenames))
        self.trainval_filenames = self.filenames[:split]
        self.test_filenames = self.filenames[split:]
        self.trainval_filenames.sort()
        self.test_filenames.sort()

    def output(self):
        with open(join(self.output_dir,'trainval.txt'), 'w') as f:
            for filename in self.trainval_filenames:
                f.write(splitext(filename)[0] + '\n')
        f.close()

        with open(join(self.output_dir,'test.txt'), 'w') as f:
            for filename in self.test_filenames:
                f.write(splitext(filename)[0] + '\n')
        f.close()

if __name__ == "__main__":
    
    ##### Configure parameters here
    data_dir = 'data'
    image_dir = join(data_dir,'JPEGImages')
    output_dir = join(data_dir,'ImageSets/Main')
    trainval_ratio = 0.9
    seed_int = 1

    gen = TxtGenerator(image_dir, output_dir, trainval_ratio, seed_int)
    gen.read_images()
    gen.split()
    gen.output()