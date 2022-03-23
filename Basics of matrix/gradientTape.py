import tensorflow as tf

x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
gradientOfYwrtX = tape.gradient(y, x)
print(gradientOfYwrtX)