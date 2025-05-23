module Helpers
  LABEL_MAP = [
    "t_shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot"
  ]

  MISSING_LABEL_ERROR = <<~TEXT
  label `%s` not found.
  Available labels:
  #{LABEL_MAP.join("\n")}
  TEXT

  def self.save_image(tensor, path)
    pixels = tensor.squeeze.mul(255).type(:uint8).to_a

    image = ChunkyPNG::Image.new(28, 28, ChunkyPNG::Color.grayscale(0))

    28.times do |y|
      28.times do |x|
        grey = pixels[y][x]
        color = ChunkyPNG::Color.grayscale(grey)
        image[x, y] = color
      end
    end

    image.save(path)
  end

  def self.image_by_label(dataset:, label:)
    label_id = LABEL_MAP.index(label)

    raise ArgumentError, MISSING_LABEL_ERROR % label if label_id.nil?

    dataset.size.times do |i|
      image, label = dataset[i]
      return image if label == label_id
    end
  end

  def self.random_image_by_label(dataset:, label:)
    label_id = LABEL_MAP.index(label)

    raise ArgumentError, MISSING_LABEL_ERROR % label if label_id.nil?

    loop do
      i = rand(dataset.size)
      image, label = dataset[i]
      return image if label == label_id
    end
  end

  def self.run_id
    ('a'..'z').to_a.sample(6).join
  end
end
