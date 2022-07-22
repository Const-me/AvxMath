#ifdef _MSC_VER
#include "testDx.h"
#endif
#include "testStdlib.h"

int main()
{
#ifdef _MSC_VER
	testDx();
#endif
	testStdlib();
	return 0;
}