require "byebug"
require "torch"
require "torchvision"
require_relative "autoencoder"

Torch.manual_seed(1)
device = Torch.device("mps")

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-8

transform = TorchVision::Transforms::Compose.new([
  TorchVision::Transforms::ToTensor.new,
  TorchVision::Transforms::Normalize.new([0.5], [0.5]),
])

train_dataset = TorchVision::Datasets::FashionMNIST.new(
  "./data", train: true, download: true, transform: transform
)

train_loader = Torch::Utils::Data::DataLoader.new(
  train_dataset, batch_size: BATCH_SIZE, shuffle: true
)

model = AutoEncoder.new
model.to(device)

criterion = Torch::NN::MSELoss.new
optimizer = Torch::Optim::Adam.new(
  model.parameters,
  lr: LEARNING_RATE,
  weight_decay: WEIGHT_DECAY
)

losses = []

EPOCHS.times do |epoch|
  model.train
  total_loss = 0.0

  train_loader.each_with_index do |(images, _labels), batch_idx|
    images = images.to(device)

    optimizer.zero_grad

    output = model.call(images)
    loss = criterion.call(output, images)

    loss.backward
    optimizer.step

    total_loss += loss.item

    losses << loss.item

    if batch_idx % 100 == 0
      puts "Epoch #{epoch + 1} Batch #{batch_idx} Loss #{'%.4f' % loss.item}"
    end
  end

  avg_loss = total_loss / train_loader.size
  puts "Epoch #{epoch + 1} complete. Average Loss: #{avg_loss}"
end

File.open("tmp/losses.csv", "w") do |file|
  file.puts "Iteration,Loss"

  losses.each_with_index do |loss, itteration|
    file.puts "#{itteration},#{loss}"
  end
end

Torch.save(model.state_dict, "models/autoencoder.pt")
