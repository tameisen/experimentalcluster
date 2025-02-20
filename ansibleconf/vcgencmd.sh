#!/bin/bash
sudo vcgencmd pmic_read_adc | sed -r 's/([A-Z0-9_]+)_[A-Z]+\\s(current|volt)\\([0-9]+\\)=([0-9.]+)(A|V)/\\1_\\2=\\3/g'
