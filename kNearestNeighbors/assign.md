K-Nearest Neighbors
=

Overview
--------

In this homework you'll implement a k-Nearest Neighbor classification
framework to take a drawing of a number and output what number it
corresponds to.

![Samples of Handwritten Digits](mnist.png "MNIST Digits")

This is designed to be a *very easy* homework.  If you're spending a
lot of time on this assignment, you are either:

* not prepared to take the course (i.e., if you're struggling with Python)
* seriously over-thinking the assignment
* trying to reimplement too much of the assignment

Most of this assignment will be done by calling libraries that have
already been implemented for you.  If you are implementing
n-dimensional search or a median algorithm, you are generating extra
work for yourself and making yourself vulnerable to errors.

You'll turn in your code on ELMS.  This assignment is worth 15
points.  Other assignments will be worth more.

What you have to do
----

Coding (10 points):

1.  (Optional) Store necessary data in the constructor so you can do classification later.
1.  Modify the _majority_ function so that it returns the *value* associated with the most *indices*.
1.  Modify the _classify_ function so that it finds the closest indicies to the query point.
1.  Modify the _confusion matrix_ function to classify examples and record which number it got right.

Analysis (5 points):

1.  What is the role of the number of training points to accuracy?
1.  What is the role of _k_ to accuracy?
1.  What numbers get confused with each other most easily?

What to turn in
-

1.  Submit your _knn.py_ file
1.  Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)

Unit Tests
=

I've provided unit tests based on the example that we worked through
in class.  Before running your code on read data, make sure it passes
all of the unit tests.

```
cs244-33-dhcp:knn jbg$ python tests.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.003s

OK
```

Initially, it will fail all of them:
```
cs244-33-dhcp:knn jbg$ python tests.py
FFF
======================================================================
FAIL: test1 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 20, in test1
    self.assertEqual(self.knn[1].classify(self.queries[1]), -1)
AssertionError: 1 != -1

======================================================================
FAIL: test2 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 25, in test2
    self.assertEqual(self.knn[2].classify(self.queries[0]), 1)
AssertionError: -1 != 1

======================================================================
FAIL: test3 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 31, in test3
    self.assertEqual(self.knn[3].classify(self.queries[0]), 1)
AssertionError: -1 != 1

----------------------------------------------------------------------
Ran 3 tests in 0.002s

FAILED (failures=3)
```