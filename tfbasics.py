import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

# result = x1 * x2  # This works, but it is slow
result = tf.multiply(x1, x2)  # Abstract Tensor in computation graph. Usually use matmul()
print("Result:", result)

# sess = tf.Session()
# print(sess.run(result))  # Computations run here
# sess.close()

with tf.Session() as sess:  # Automatically closes sess when done
    output = sess.run(result)  # Save in Python variable
    print(output)  # Computations run here
