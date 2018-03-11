class TrueSkillStatistic:

    def get_error(self, image1data, image2data):
        z = 0  # correct non rain forecast
        h = 0  # hits
        m = 0  # misses
        f = 0  # false alarms
        zipped = list(zip(image1data, image2data))
        for pair in zipped:
            z += 1 if (pair[0] == 0 and pair[1] == 0) else 0
            h += 1 if (pair[0] > 0 and pair[1] > 0) else 0
            m += 1 if (pair[1] > 0 and pair[0] == 0) else 0
            f += 1 if (pair[1] == 0 and pair[0] > 0) else 0
        z_f = 1 if (z + f == 0) else z + f
        m_h = 1 if (m + h == 0) else m + h
        score = ((z*h-f*m)/(z_f * m_h))
        return score

