// Common/image.cpp
#include "image.h"

#include <cassert>
#include <cstdlib>
#include <exception>

Image::Image() : Image(0, 0, 0, nullptr) {}

Image::Image(unsigned M, unsigned N, unsigned Q)
    : Image(M, N, Q, new Image::pixelT[M * N]) {}

Image::Image(const Image& oldImage) : Image(oldImage.M, oldImage.N, oldImage.Q) {
	for (unsigned i = 0; i < M * N; i++) { pixelValue[i] = oldImage.pixelValue[i]; }
}

// Move constructor - take old image's pixel values and make old image invalid
Image::Image(Image&& oldImage)
    : Image(oldImage.M, oldImage.N, oldImage.Q, oldImage.pixelValue) {
	oldImage.M = oldImage.N = oldImage.Q = 0;
	oldImage.pixelValue                  = nullptr;
}

Image::Image(unsigned M, unsigned N, unsigned Q, pixelT* pixels)
    : M(M), N(N), Q(Q), pixelValue(pixels) {}

Image::~Image() {
	if (pixelValue != nullptr) { delete[] pixelValue; }
}

// Slightly modified version of readImage() function provided by Dr. Bebis
Image Image::read(std::istream& in) {
	int N, M, Q;
	unsigned char* charImage;
	char header[100], *ptr;

	static_assert(sizeof(Image::pixelT) == 1,
	              "Image reading only supported for single-byte pixel types.");

	// read header
	in.getline(header, 100, '\n');
	if ((header[0] != 'P') || (header[1] != '5')) {
		throw std::runtime_error("Image is not PGM!");
	}

	in.getline(header, 100, '\n');
	while (header[0] == '#') in.getline(header, 100, '\n');

	N = strtol(header, &ptr, 0);
	M = atoi(ptr);

	in.getline(header, 100, '\n');
	Q = strtol(header, &ptr, 0);

	if (Q > 255)
		throw std::runtime_error("Image cannot be read correctly (Q > 255)!");

	charImage = new unsigned char[M * N];

	in.read(reinterpret_cast<char*>(charImage), (M * N) * sizeof(unsigned char));

	if (in.fail()) throw std::runtime_error("Image has wrong size!");

	return Image(M, N, Q, charImage);
}

// Slightly modified version of writeImage() function provided by Dr. Bebis
std::ostream& operator<<(std::ostream& out, const Image& im) {
	static_assert(sizeof(Image::pixelT) == 1,
	              "Image writing only supported for single-byte pixel types.");

	out << "P5" << std::endl;
	out << im.N << " " << im.M << std::endl;
	out << im.Q << std::endl;

	out.write(reinterpret_cast<char*>(im.pixelValue),
	          (im.M * im.N) * sizeof(unsigned char));

	if (out.fail()) throw std::runtime_error("Something failed with writing image.");
}

Image& Image::operator=(const Image& rhs) {
	if (pixelValue != nullptr) delete[] pixelValue;

	M = rhs.M;
	N = rhs.N;
	Q = rhs.Q;

	pixelValue = new pixelT[M * N];

	for (unsigned i = 0; i < M * N; i++) pixelValue[i] = rhs.pixelValue[i];

	return *this;
}

Image& Image::operator=(Image&& rhs) {
	if (pixelValue != nullptr) delete[] pixelValue;

	M          = rhs.M;
	N          = rhs.N;
	Q          = rhs.Q;
	pixelValue = rhs.pixelValue;

	rhs.M = rhs.N = rhs.Q = 0;
	rhs.pixelValue        = nullptr;

	return *this;
}

Image::pixelT* Image::operator[](unsigned i) {
	return pixelValue + i * N;
}

const Image::pixelT* Image::operator[](unsigned i) const {
	return pixelValue + i * N;
}

// Slightly modified version of readImageHeader() function provided by Dr. Bebis
Image::Header Image::Header::read(std::istream& in) {
	unsigned char* charImage;
	char header[100], *ptr;
	Header re;

	// read header
	in.getline(header, 100, '\n');
	if ((header[0] == 'P') && (header[1] == '5')) {
		re.type = GRAY;
	} else if ((header[0] == 'P') && (header[1] == '6')) {
		re.type = COLOR;
	} else
		throw std::runtime_error("Image is not PGM or PPM!");

	in.getline(header, 100, '\n');
	while (header[0] == '#') in.getline(header, 100, '\n');

	re.N = strtol(header, &ptr, 0);
	re.M = atoi(ptr);

	in.getline(header, 100, '\n');

	re.Q = strtol(header, &ptr, 0);

	return re;
}

std::ostream& operator<<(std::ostream& out, const Image::Header& head) {
	switch (head.type) {
		case Image::Header::Type::COLOR:
			out << "PPM Color ";
			break;
		case Image::Header::Type::GRAY:
			out << "PGM Grayscale ";
	}
	out << "Image size " << head.M << " x " << head.N << " and max value of "
	    << head.Q << ".";
}