# CSE REU 2018

**_Parsing data using OpenCL_**

The data being parsed is comma-seperated with delimited zones within which
characters follow special rules. To efficiently parse this data, a
parallel implemetation was used. The function that determines if a
charater is in a delimited zone is defined such that it is associative.
Thus, this parallel parsing implementation uses a parallel scan as
dedscribed in the following slides and paper:

http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf
https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf

**_Contributors:_**

Gabe Maayan

Jabari Booker