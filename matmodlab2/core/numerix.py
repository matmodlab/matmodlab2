import numpy as np


def amag(a):
    return np.sqrt(np.sum(a * a))


def rms_error(t1, d1, t2, d2, disp=1):
    """Compute the RMS and normalized RMS error"""
    t1 = np.asarray(t1)
    d1 = np.asarray(d1)
    t2 = np.asarray(t2)
    d2 = np.asarray(d2)

    if t1.shape[0] == t2.shape[0]:
        rms = np.sqrt(np.mean((d1 - d2) ** 2))
    else:
        rms = interp_rms_error(t1, d1, t2, d2)
    dnom = np.amax(np.abs(d1))
    if dnom < 1.0e-12:
        dnom = 1.0
    if disp:
        return rms, rms / dnom
    return rms / dnom


def interp_rms_error(t1, d1, t2, d2):
    """Compute RMS error by interpolation"""
    ti = max(np.amin(t1), np.amin(t2))
    tf = min(np.amax(t1), np.amax(t2))
    n = t1.shape[0]
    f1 = lambda x: np.interp(x, t1, d1)
    f2 = lambda x: np.interp(x, t2, d2)
    rms = np.sqrt(np.mean([(f1(t) - f2(t)) ** 2 for t in np.linspace(ti, tf, n)]))
    return rms


def line_intersect(pt0, pt1, pt2, pt3):
    """Determine if and where two line segments intersect.

    The four (x,y) point pairs define the two line segments with
    pt0 and pt1 defining the first segment and pt2 and pt3 defining
    the second.

    This function returns 'bool, list' with 'bool' being true if the
    line segments intersect and 'list' being the (x,y) coordinates of
    the intersection point. If no intersection point is found, 'False, []'
    is returned.

    Created: 2015 April 29, msswan
    """
    # Simplest check: do bounding boxes intersect?
    if max(pt0[0], pt1[0]) < min(pt2[0], pt3[0]) or max(  # box 0 left of box 1
        pt2[0], pt3[0]
    ) < min(
        pt0[0], pt1[0]
    ):  # box 1 left of box 0
        return False, []
    elif max(pt0[1], pt1[1]) < min(pt2[1], pt3[1]) or max(  # box 0 below box 1
        pt2[1], pt3[1]
    ) < min(
        pt0[1], pt1[1]
    ):  # box 1 below box 0
        return False, []

    p0 = np.array(pt0)
    p1 = np.array(pt1)
    p2 = np.array(pt2)
    p3 = np.array(pt3)

    # If both points of the other segment are on one side then they do
    # not intersect.

    # Check from segment 1's perspective
    cp0 = np.cross(p1 - p0, p2 - p0)
    cp1 = np.cross(p1 - p0, p3 - p0)
    if cp0 * cp1 > 0.0:
        return False, []

    # Check from segment 2's perspective
    cp0 = np.cross(p3 - p2, p0 - p2)
    cp1 = np.cross(p3 - p2, p1 - p2)
    if cp0 * cp1 > 0.0:
        return False, []

    # If we've gotten this far then they intersect.
    denom = np.cross(p1 - p0, p3 - p2)
    numer = np.cross(p2 - p0, p1 - p0)

    if denom == 0.0 and numer == 0.0:
        # points are colinear, take point closest to p0
        if np.linalg.norm(p2 - p0) < np.linalg.norm(p3 - p0):
            return True, p2
        else:
            return True, p3

    u = numer / denom
    return True, p2 + u * (p3 - p2)


def get_area(ptlist):
    """Calculate the area of a polygon defined by a list of points.

    The variable ptlist is a list of (x, y) point pairs. Be careful,
    the implementation can give unexpected results with self-intersecting
    polygons.

    The output will always be non-negative.

    Created: 2015 April 29, msswan
    """
    I = lambda pt1, pt2: (pt2[1] + pt1[1]) * (pt2[0] - pt1[0]) / 2.0
    area = I(ptlist[-1], ptlist[0])
    for idx in range(0, len(ptlist) - 1):
        area += I(ptlist[idx], ptlist[idx + 1])
    return abs(area)


def calculate_bounded_area(x0, y0, x1, y1):
    """Calculate the area bounded by two potentially-nonmonotonic 2D data sets

    This function is written to calculate the area between two arbitrary
    piecewise-linear curves. The method was inspired by the arbitrary polygon
    filling routines in vector software programs when the polygon
    self-intersects.

    Created: 2015 April 29, msswan
    """

    # We start by taking the start of the first data set (pts0) and loop over
    # each segment (starting with the closest) and check to see if the
    # second data (pts1) set intersects. If there is an intersection, it joins
    # all the points together to make a polygon (reversing pts1 so that the
    # polygon integration calculation goes around in a single direction) and
    # calculates the area from that. Now it removes the points that it used to
    # create the polygon and adds the intersection point to pts0 (which is the
    # new starting point) and starts the loop again.

    # Turn the data into lists of tuples (x,y) coordinates
    pts0 = list(zip(x0, y0))
    pts1 = list(zip(x1, y1))

    area = 0.0
    while len(pts0) + len(pts1) > 0:
        shouldbreak = False
        for idx in range(0, len(pts0) - 1):
            for jdx in range(0, len(pts1) - 1):
                doesintersect, int_pt = line_intersect(
                    pts0[idx], pts0[idx + 1], pts1[jdx], pts1[jdx + 1]
                )
                if not doesintersect:
                    continue

                polygon = (
                    list(reversed(pts1[:jdx]))
                    + pts0[:idx]
                    + [
                        int_pt,
                    ]
                )
                area += get_area(polygon)

                # Trim the processed points off of the datasets
                pts0 = [
                    int_pt,
                ] + pts0[idx + 1 :]
                pts1 = pts1[jdx + 1 :]

                # Exit out of both for-loops
                shouldbreak = True
                break

            if shouldbreak:
                break
        else:
            # Make a polygon out of whatever points remain
            polygon = list(reversed(pts1)) + pts0
            area += get_area(polygon)

            # exit the while loop
            break

    return area


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if True:
        # colinear but not intersecting
        pt0 = np.array([0.0, 0.0])
        pt1 = np.array([1.0, 1.0])
        pt2 = np.array([3.0, 3.0])
        pt3 = np.array([4.0, 4.0])

        a = line_intersect(pt0, pt1, pt2, pt3)
        print("colinear but not intersecting:")
        print(a)

        # colinear but not intersecting
        pt0 = np.array([0.0, 0.0])
        pt1 = np.array([2.0, 2.0])
        pt2 = np.array([1.0, 1.0])
        pt3 = np.array([4.0, 4.0])

        a = line_intersect(pt0, pt1, pt2, pt3)
        print("colinear and intersecting:")
        print(a)

    if True:
        # Do visual tests on if line segments intersect
        for idx in range(0, 10):
            pt0 = np.random.uniform(-1.0, 1.0, 2)
            pt1 = np.random.uniform(-1.0, 1.0, 2)
            pt2 = np.random.uniform(-1.0, 1.0, 2)
            pt3 = np.random.uniform(-1.0, 1.0, 2)

            a = line_intersect(pt0, pt1, pt2, pt3)

            plt.clf()
            plt.cla()
            style = "r-o" if a[0] else "g-o"
            x, y = zip(pt0, pt1)
            plt.plot(x, y, style)
            x, y = zip(pt2, pt3)
            plt.plot(x, y, style)

            if a[0]:
                plt.plot(
                    [
                        a[1][0],
                    ],
                    [
                        a[1][1],
                    ],
                    "b-o",
                    lw=10,
                )

            plt.suptitle(repr(a[0]))
            plt.show()

    if True:
        # Test the bounded_area() function with quadratic datasets with noise
        x0 = np.linspace(0.0, 1.0, 1000)
        y0 = x0 * x0
        x0 = x0 + np.random.normal(0.0, 0.02, len(x0))
        y0 = y0 + np.random.normal(0.0, 0.02, len(x0))

        x1 = np.linspace(0.0, 1.0, 2000)
        y1 = x1 * x1
        x1 = x1 + np.random.normal(0.0, 0.02, len(x1))
        y1 = y1 + np.random.normal(0.0, 0.02, len(x1))

        print("area", calculate_bounded_area(x0, y0, x1, y1))

        plt.clf()
        plt.cla()
        plt.plot(x0, y0, "r-")
        plt.plot(x1, y1, "g-")
        plt.show()
