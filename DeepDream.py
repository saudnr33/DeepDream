



'''

Utils: all util functions

'''
lower_image_bound = (-torch.tensor(imagenet_mean) / torch.tensor(imagenet_std)).reshape(1, -1, 1, 1).to(device)
upper_image_bound = ((1 - torch.tensor(imagenet_mean)) / torch.tensor(imagenet_std)).reshape(1, -1, 1, 1).to(device)

def gradient_ascent(
    input_tensor,
    model,
    layers_to_use,
    num_gradient_ascent_iterations = 1,
    learning_rate = 0.01 #
):


    input = Variable(input_tensor, requires_grad=True)



    for i in range(num_gradient_ascent_iterations):
          model.zero_grad()
          image_features = get_features(input, model, layers_to_use)
          losses = [(transforms.functional.normalize(image_features[layers_to_use[key]], 0.5, 0.5) ).norm() for key in layers_to_use]
          image_features = sum(losses)/len(losses)
          image_features.backward()


          grad = input.grad.data

          avg_grad = torch.mean(grad)
          input.data += learning_rate *grad/torch.abs(avg_grad)
          input.data = torch.clamp(input.data, lower_image_bound, upper_image_bound)
          input.grad.data.zero_()



    return input.data


def deep_dream(image,
               model,
               layers_to_use,

               octave_scale = 1,
               pyramid_levels = 1,
               image_size = 600,
               spatial_shift_size = (32, 32),
               num_gradient_ascent_iterations = 1,
               lr = 0.001
              ):
    if isinstance(image, str):
        image_tensor = load_image(image, max_size=image_size).to(device)
    elif isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
    else:
        image_tensor = image

    input_shape = image_tensor.shape[2:]  # save initial height and width

    imageProcessed = image

    detail = torch.zeros_like(transforms.functional.resize(imageProcessed, (int(input_shape[0]*octave_scale**(pyramid_levels - 1)) , int(input_shape[1]*octave_scale**(pyramid_levels- 1)) )))
    for scale_level in range(pyramid_levels - 1, -1, -1):

        ## Resample the image tensor by the factor defined by octave_scale to obtain a version resampled to the current level
        octave = transforms.functional.resize(imageProcessed, (int(input_shape[0]/(octave_scale**scale_level)) , int(input_shape[1]/(octave_scale**scale_level)) ))


        detail = transforms.functional.resize(detail, (octave.size()[2] , octave.size()[3]) )
        input_image = octave + detail

            # detail = transforms.functional.resize(detail, (octave.size()[2] , octave.size()[3]) )
            # input_image = octave + spatial_shift(detail,spatial_shift_size[0], spatial_shift_size[1])

        input_image = spatial_shift(input_image,spatial_shift_size[0], spatial_shift_size[1])
        imageProcessed = gradient_ascent(input_image, model, layers_to_use, num_gradient_ascent_iterations = num_gradient_ascent_iterations, learning_rate = lr)
        imageProcessed = spatial_shift(imageProcessed, -spatial_shift_size[0], -spatial_shift_size[1])
            # show_image(imageProcessed, figsize=(12, 8))



        detail = imageProcessed - octave





    return imageProcessed

def get_features(image, model, layers):
    # Run an image forward through a model and get the features for a set of layers.

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)

        if name in layers:
            features[layers[name]] = x

    return features
def spatial_shift(tensor, h_shift, w_shift):
    shifted = torch.roll(tensor, h_shift, 3)
    shifted = torch.roll(shifted, w_shift,2)
    return shifted


def load_image(img_path, max_size = 400, shape = None):
    ''' Load and downscale an image if the shorter side is longer than <max_size> px '''

    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    if min(image.size) > max_size:
        size = max_size
    else:
        size = min(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize(imagenet_mean,
                                             imagenet_std)])

    # discard alpha channel (:3) and append the batch dimension (unsqueeze(0))
    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image
def im_convert(tensor, mean=imagenet_mean, std=imagenet_std):
    ''' Convert a PyTorch tensor to a NumPy image. '''

    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array(std) + np.array(mean)
    image = image.clip(0, 1)

    return image

    def show_image(image, figsize=(16, 10)):
    if torch.is_tensor(image):
        image = im_convert(image)

    fig = plt.figure(figsize=figsize, dpi=100)  # otherwise plots are really small in Jupyter Notebook
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_image(image, image_path):
    image = im_convert(image)

    formatted = (image * 255 / np.max(image)).astype('uint8')
    pil_image = Image.fromarray(formatted, 'RGB')
    pil_image.save(image_path, 'JPEG', quality=95)

def format_time(seconds):
    time_str = ''
    hrs = seconds // 3600
    time_str += f'{hrs:.0f}hrs ' if hrs > 0 else ''
    mins = (seconds % 3600) // 60
    time_str += f'{mins:.0f}min ' if mins > 0 else ''
    time_str += f'{(seconds % 60):.2f}sec'
    return time_str
