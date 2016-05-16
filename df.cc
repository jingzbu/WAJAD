#include <iostream>
#include <string>
#include "df.h"
using namespace std;

int main()
{
  float d;
  d = distance(42.359239, -71.11617, 42.3387, -71.104, 'K');
  cout << d << endl;
}
