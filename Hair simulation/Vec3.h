#pragma once
#include <math.h>

template <typename T>
class Vec3
{
public:
	T		m_dX;
	T		m_dY;
	T		m_dZ;
public:
	Vec3(void)
	{
		m_dX = (T)0;
		m_dY = (T)0;
		m_dZ = (T)0;
	}
	Vec3(Vec3 const &v)
	{
		m_dX = v.m_dX;
		m_dY = v.m_dY;
		m_dZ = v.m_dZ;
	}
	Vec3(T const &x, T const &y, T const &z)
	{
		m_dX = x;
		m_dY = y;
		m_dZ = z;
	}
	~Vec3() {}
	void						x(T d) { m_dX = d; }
	void						y(T d) { m_dY = d; }
	void						z(T d) { m_dZ = d; }
	void						SetX(T d) { m_dX = d; }
	void						SetY(T d) { m_dY = d; }
	void						SetZ(T d) { m_dZ = d; }
	void						Set(Vec3<T> &v)
	{
		m_dX = v.m_dX;
		m_dY = v.m_dY;
		m_dZ = v.m_dZ;
	}
	void						Set(T x)
	{
		m_dX = x;
		m_dY = x;
		m_dZ = x;
	}
	void						Set(T x, T y, T z)
	{
		m_dX = x;
		m_dY = y;
		m_dZ = z;
	}
	void						Clear(void)
	{
		m_dX = 0.0;
		m_dY = 0.0;
		m_dZ = 0.0;
	}
	void						Print(void)
	{
		printf("%f, %f, %f\n", m_dX, m_dY, m_dZ);
	}
	void						Normalize(void)
	{
		double norm = GetNorm();
		if (norm != 0) {
			m_dX = m_dX / norm;
			m_dY = m_dY / norm;
			m_dZ = m_dZ / norm;
		}
	}
	void						Inverse(void)
	{
		m_dX *= -1.0;
		m_dY *= -1.0;
		m_dZ *= -1.0;
	}
	T						GetNorm(void)
	{
		return sqrt(m_dX*m_dX + m_dY * m_dY + m_dZ * m_dZ);
	}
	T				double SQRT(double input, double x)
{
      for(int i=0; i<10; i++) {  x = (x + (input / x)) / 2 ; }    // 이 한줄 땜에 주말내내 머리에 쥐남 ㅠㅠ

 return x ;
}		Length(void)
	{
		return(T)(sqrt(LengthSquared()));
	}
	T						LengthSquared(void)
	{
		return (T)(m_dX*m_dX + m_dY * m_dY + m_dZ * m_dZ);
	}
	T						Dot(Vec3<T>& v)
	{
		return (m_dX*v.GetX() + m_dY * v.GetY() + m_dZ * v.GetZ());
	}
	T						x(void) { return m_dX; }
	T						y(void) { return m_dY; }
	T						z(void) { return m_dZ; }
	T						GetX(void) { return m_dX; }
	T						GetY(void) { return m_dY; }
	T						GetZ(void) { return m_dZ; }
	T						&Get(int n)
	{
		return  *((&m_dX) + n);
	}
	Vec3<T>	Cross(Vec3<T>& v)
	{
		Vec3<T> vector;
		vector.SetX((m_dY*v.GetZ()) - (m_dZ*v.GetY()));
		vector.SetY((m_dZ*v.GetX()) - (m_dX*v.GetZ()));
		vector.SetZ((m_dX*v.GetY()) - (m_dY*v.GetX()));
		return vector;
	}
public:
	bool						operator==(Vec3 const &v) const
	{
		return m_dX == v.m_dX && m_dY == v.m_dY && m_dZ == v.m_dZ;
	}
	bool						operator!=(Vec3 const & v) const
	{
		return m_dX != v.m_dX && m_dY != v.m_dY && m_dZ != v.m_dZ;
	}
	T						&operator()(int index)
	{
		return  *((&m_dX) + index);
	}
	T						&operator[](int index)
	{
		return  *((&m_dX) + index);
	}
	T const					&operator()(int index) const
	{
		return  *((&m_dX) + index);
	}
	T const					&operator[](int index) const
	{
		return  *((&m_dX) + index);
	}
	Vec3<T>	Get_Ortho(void)
	{
		if (m_dX != 0) {
			return Vec3(-m_dY, m_dX, 0);
		}
		else {
			return Vec3(1, 0, 0);
		}
	}
	Vec3<T>	&operator=(Vec3 const &v)
	{
		m_dX = v.m_dX;
		m_dY = v.m_dY;
		m_dZ = v.m_dZ;
		return *this;
	}
	Vec3<T>	&operator+=(Vec3 const &v)
	{
		m_dX += v.m_dX;
		m_dY += v.m_dY;
		m_dZ += v.m_dZ;
		return *this;
	}
	Vec3<T>	&operator+=(T v)
	{
		m_dX += v;
		m_dY += v;
		m_dZ += v;
		return *this;
	}
	Vec3<T>	&operator-=(T v)
	{
		m_dX -= v;
		m_dY -= v;
		m_dZ -= v;
		return *this;
	}
	Vec3<T>	&operator-=(Vec3 const &v)
	{
		m_dX -= v.m_dX;
		m_dY -= v.m_dY;
		m_dZ -= v.m_dZ;
		return *this;
	}
	Vec3<T>	&operator*=(T const &d)
	{
		m_dX *= d;
		m_dY *= d;
		m_dZ *= d;
		return *this;
	}
	Vec3<T>	&operator*=(Vec3<T> &v)
	{
		m_dX *= v.x();
		m_dY *= v.y();
		m_dZ *= v.z();
		return *this;
	}
	Vec3<T>	&operator/=(T const &d)
	{
		m_dX /= d;
		m_dY /= d;
		m_dZ /= d;
		return *this;
	}
	Vec3<T>	operator/(const T &d)
	{
		return Vec3(m_dX / d, m_dY / d, m_dZ / d);
	}
	Vec3<T>	operator*(const T &d)
	{
		return Vec3(m_dX*d, m_dY*d, m_dZ*d);
	}
	Vec3<T>	operator-(const T &d)
	{
		return Vec3(m_dX - d, m_dY - d, m_dZ - d);
	}
	Vec3<T>	operator+(const T &d)
	{
		return Vec3(m_dX + d, m_dY + d, m_dZ + d);
	}
	Vec3<T>	operator-() const
	{
		return Vec3(-m_dX, -m_dY, -m_dZ);
	}
	Vec3<T>	operator+(Vec3 const &v) const
	{
		return Vec3(m_dX + v.m_dX, m_dY + v.m_dY, m_dZ + v.m_dZ);
	}
	Vec3<T>	operator-(Vec3 const &v) const
	{
		return Vec3(m_dX - v.m_dX, m_dY - v.m_dY, m_dZ - v.m_dZ);
	}
	Vec3<T>	operator*(Vec3<T>	&v)
	{
		return Vec3(m_dX*v.m_dX, m_dY*v.m_dY, m_dZ*v.m_dZ);
	}
	T						AngleBetweenVectors(Vec3<T> &v)
	{
		T dot, cross;
		Vec3<T> tmp;
		dot = Dot(v);  // Cos * norms
		tmp.Set(Cross(v));
		cross = tmp.Length(); //  Sin * norms
		return atan2(cross, dot);
	}
	void						ProcessInner(Vec3<T> *u, Vec3<T> *v)
	{
		m_dX = u->y()*v->z() - u->z()*v->y();
		m_dY = u->z()*v->x() - u->x()*v->z();
		m_dZ = u->x()*v->y() - u->y()*v->x();
	}
	void						Debugging_trace_glVertex(void)
	{
		TRACE("glVertex3f(%f,%f,%f);\n", m_dX, m_dY, m_dZ);
	}

};