#include <cmath>
class random {
private:
	int a = 789;
	int a_1 = a * 10;
	int c = 123;
	int c_1 = c * 40;
	int m = 481;
	int m_1 = m * 2;
	int next = 789;
public:
	int rand(bool standart)
	{
		if (standart == true) {
			next = ((a * next) + c) % m;
		}
		else
		{
			next = int((a * next) + pow(next + c, 2)) % m;
		}
		m = ((next + a_1) * c_1) % a + next + m_1;
		c = ((m * next) + c) % a;
		return next;
	}

	void srand(int seed, int new_a, int new_c, int new_m)
	{
		next = seed;
		a = new_a;
		c = new_c;
		m = new_m;
	}
};