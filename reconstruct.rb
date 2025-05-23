require "byebug"
require "chunky_png"
require "torch"
require "torchvision"
require_relative "autoencoder"
require_relative "helpers"

transform = TorchVision::Transforms::Compose.new([
  TorchVision::Transforms::ToTensor.new
])

dataset = TorchVision::Datasets::FashionMNIST.new(
  "./data", train: false, download: true, transform: transform
)

model = AutoEncoder.new
model.load_state_dict(Torch.load("models/autoencoder.pt"))
model.eval

label = ARGV[0] || raise(ArgumentError, "Label not provided")

image = Helpers.random_image_by_label(dataset: dataset, label: label)

run_id = Helpers.run_id

Torch.no_grad do
  reconstructed = model.forward(image)

  Helpers.save_image(image, "tmp/#{run_id}--original_#{label}.png")
  Helpers.save_image(reconstructed, "tmp/#{run_id}--reconstructed_#{label}.png")
end

puts run_id
