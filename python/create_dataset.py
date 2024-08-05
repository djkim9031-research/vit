import os
from torchvision import datasets

def download_and_save_as_bmp(isTrainingSet):
    output_dir = ''
    if(isTrainingSet):
        output_dir += '../data/train'
    else:
        output_dir += '../data/test'

    os.makedirs(output_dir, exist_ok=True)

    labels_file_path = os.path.join(output_dir, 'labels.txt')
    cifar10_data = datasets.CIFAR10(root='.', train=isTrainingSet, download=True)

    with open(labels_file_path, 'w') as labels_file:
        for idx, (image, label) in enumerate(cifar10_data):
            image_path = os.path.join(output_dir, f'{idx:05d}.bmp')
            image.save(image_path)

            labels_file.write(f'{idx:05d}: {label}\n')

            if idx % 1000 == 0:
                print(f'Saved {idx}-th data.')
    print('All dataset saved.')


download_and_save_as_bmp(True) # train dataset 
download_and_save_as_bmp(False) # Test dataset