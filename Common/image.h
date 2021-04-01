// Common/image.h
#pragma once

#include <iostream>

class Image {
public:
	// The type that is used for the value of each pixel
	// As of right now, read and operator<< only work if it is one byte large
	typedef unsigned char pixelT;
	// Struct for reading just the header of an image
	struct Header {
		enum Type {
			COLOR,
			GRAY,
		} type;

		unsigned M, N, Q;

		// Read header from file
		// Throws std::runtime_error for any errors encountered,
		// such as not having a valid PGM/PPM header
		static Header read(std::istream &in);
	};

	Image();
	Image(unsigned M, unsigned N, unsigned Q, pixelT *p);
	Image(unsigned M, unsigned N, unsigned Q);
	Image(const Image &);  // Copy constructor
	Image(Image &&);       // Move constructor
	~Image();

	// Read from stream (such as file)
	// Throws std::runtime_error for any errors encountered,
	// such as not being a valid PGM image
	static Image read(std::istream &in);

	// Output to stream (such as file)
	friend std::ostream &operator<<(std::ostream &out, const Image &im);

	// Pixel access - works like 2D array i.e. image[i][j]
	pixelT *operator[](unsigned i);
	const pixelT *operator[](unsigned i) const;
	Image &operator=(const Image &rhs);  // Assignment
	Image &operator=(Image &&rhs);       // Move

	// Read-only properties
	pixelT *const &pixels  = pixelValue;
	const unsigned &rows   = M;
	const unsigned &cols   = N;
	const unsigned &maxVal = Q;

private:
	unsigned M, N, Q;
	pixelT *pixelValue;
};

std::ostream &operator<<(std::ostream &out, const Image::Header &head);