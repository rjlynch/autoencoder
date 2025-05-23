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

label_a = ARGV[0] || raise(ArgumentError, "Label A not provided")
label_b = ARGV[1] || raise(ArgumentError, "Label B not provided")

image_a = Helpers.random_image_by_label(dataset: dataset, label: label_a)
image_b = Helpers.random_image_by_label(dataset: dataset, label: label_b)

run_id = Helpers.run_id

Torch.no_grad do
  latent_a = model.forward_encoder(image_a)
  latent_b = model.forward_encoder(image_b)

  combined_latent = (latent_a + latent_b) / 2.0

  combined_image = model.forward_decoder(combined_latent)

  combined_label = ["combined", label_a, label_b].join('-')

  Helpers.save_image(image_a, "tmp/#{run_id}--#{label_a}-a.png")
  Helpers.save_image(image_b, "tmp/#{run_id}--#{label_b}-b.png")
  Helpers.save_image(combined_image, "tmp/#{run_id}--#{combined_label}.png")
end

puts run_id
