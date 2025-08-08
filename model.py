import numpy as np

class CNN:
    def __init__(self, input_length, num_classes, conv_filters, kernel_size, pool_size, hidden_units):
        self.input_length = input_length
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.hidden_units = hidden_units

        # --- Convolution Layer ---
        self.conv_W = np.random.randn(conv_filters, kernel_size) * 0.01
        self.conv_b = np.zeros((conv_filters,))

        self.conv_output_size = input_length - kernel_size + 1
        self.pooled_len = self.conv_output_size // pool_size
        self.pool_stride = pool_size  # no overlap

        # --- Fully Connected Hidden Layer ---
        self.fc1_W = np.random.randn(conv_filters * self.pooled_len, hidden_units) * 0.01
        self.fc1_b = np.zeros((hidden_units,))

        # --- Output Layer ---
        self.fc2_W = np.random.randn(hidden_units, num_classes) * 0.01
        self.fc2_b = np.zeros((num_classes,))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return e_x / np.sum(e_x)

    def forward(self, x):
        # 1. Convolution
        z_conv = []
        for f in range(self.conv_filters):
            conv_out = []
            for i in range(self.conv_output_size):
                segment = x[i:i + self.kernel_size]
                value = np.dot(segment, self.conv_W[f]) + self.conv_b[f]
                conv_out.append(value)
            z_conv.append(conv_out)
        z_conv = np.array(z_conv)

        # 2. ReLU
        a_relu = self.relu(z_conv)

        # 3. Max Pooling
        pooled = []
        # Calculate the number of pooling windows (should equal self.pooled_len)
        num_windows = self.conv_output_size // self.pool_size  
        for f in range(self.conv_filters):
            pooled_f = []
            for i in range(num_windows):
                start = i * self.pool_size
                window = a_relu[f, start : start + self.pool_size]
                pooled_f.append(np.max(window))
            pooled.append(pooled_f)
        pooled = np.array(pooled)

        # 4. Flatten
        flattened = pooled.flatten()

        # 5. FC1 -> ReLU
        z_fc1 = np.dot(flattened, self.fc1_W) + self.fc1_b
        a_fc1 = self.relu(z_fc1)

        # 6. FC2 -> Softmax
        z_fc2 = np.dot(a_fc1, self.fc2_W) + self.fc2_b
        y_hat = self.softmax(z_fc2)

        # 7. Cache
        cache = {
            'x': x,
            'z_conv': z_conv,
            'a_relu': a_relu,
            'pooled': pooled,
            'flattened': flattened,
            'z_fc1': z_fc1,
            'a_fc1': a_fc1,
            'z_fc2': z_fc2,
            'y_hat': y_hat
        }

        return y_hat, cache

    def backward(self, x, y, cache):
        # 1. Output layer gradients
        d2 = cache['y_hat'] - y  # (num_classes,)

        dW2 = np.outer(cache['a_fc1'], d2)
        db2 = d2
        d1 = np.dot(self.fc2_W, d2)  # (hidden_units,)

        # 2. Hidden layer ReLU
        d1 *= self.relu_derivative(cache['z_fc1'])

        dW1 = np.outer(cache['flattened'], d1)
        db1 = d1
        dflat = np.dot(self.fc1_W, d1)
        dpool = dflat.reshape(self.conv_filters, self.pooled_len)

        # 3. Max Pooling backprop
        d_relu = np.zeros_like(cache['a_relu'])
        for f in range(self.conv_filters):
            for k in range(self.pooled_len):
                start = k * self.pool_stride
                end = start + self.pool_size
                window = cache['a_relu'][f, start:end]
                max_idx = np.argmax(window)
                d_relu[f, start + max_idx] = dpool[f, k]

        # 4. ReLU after convolution
        d_relu *= self.relu_derivative(cache['z_conv'])

        # 5. Convolution gradients
        dWconv = np.zeros_like(self.conv_W)
        dbconv = np.zeros_like(self.conv_b)
        for f in range(self.conv_filters):
            for i in range(self.conv_output_size):
                for j in range(self.kernel_size):
                    dWconv[f, j] += d_relu[f, i] * x[i + j]
            dbconv[f] = np.sum(d_relu[f])

        # Return gradients
        return {
            "fc2_W": dW2,
            "fc2_b": db2,
            "fc1_W": dW1,
            "fc1_b": db1,
            "conv_W": dWconv,
            "conv_b": dbconv
        }

    def update(self, grads, lr, clip_value=1.0):
        # Clip gradients to prevent exploding gradients
        for key in grads:
            grads[key] = np.clip(grads[key], -clip_value, clip_value)

        # Update parameters
        self.fc2_W -= lr * grads['fc2_W']
        self.fc2_b -= lr * grads['fc2_b']
        self.fc1_W -= lr * grads['fc1_W']
        self.fc1_b -= lr * grads['fc1_b']
        self.conv_W -= lr * grads['conv_W']
        self.conv_b -= lr * grads['conv_b']
