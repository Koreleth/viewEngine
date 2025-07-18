#Code by Oliver Zangenberg-Minde
# Modul: Grundlagen der Datenverarbetung
# from HS Fulda
# 18.07.2025

def m2_determinant(m):
    return m[0] * m[3] - m[1] * m[2]

def m3_determinant(m): 
    m1 = [
        m[4], m[5],
        m[7], m[8]
    ]
    m2 = [
        m[3], m[5],
        m[6], m[8]
    ]
    m3 = [
        m[3], m[4],
        m[6], m[7]
    ]

    return m[0] * m2_determinant(m1) - m[1] * m2_determinant(m2) + m[2] * m2_determinant(m3)


def m4_determinant(m):
    m1 = [
        m[5], m[6], m[7],
        m[9], m[10], m[11],
        m[13], m[14], m[15]
    ]
    m2 = [
        m[4], m[6], m[7],
        m[8], m[10], m[11],
        m[12], m[14], m[15]
    ]
    m3 = [
        m[4], m[5], m[7],
        m[8], m[9], m[11],
        m[12], m[13], m[15]
    ]
    m4 = [
        m[4], m[5], m[6],
        m[8], m[9], m[10],
        m[12], m[13], m[14]
    ]
    
    return m[0] * m3_determinant(m1) - m[1] * m3_determinant(m2) + m[2] * m3_determinant(m3) - m[3] * m3_determinant(m4)



def m4_transpose(m):
    return [
        m[0], m[4], m[8], m[12],
        m[1], m[5], m[9], m[13],
        m[2], m[6], m[10], m[14],
        m[3], m[7], m[11], m[15],
    ]



def m4_invert(m):

    # Create matrix of minors
    detM00 = m3_determinant([
        m[5], m[6], m[7],
        m[9], m[10], m[11],
        m[13], m[14], m[15]
    ])
    detM01 = m3_determinant([
        m[4], m[6], m[7],
        m[8], m[10], m[11],
        m[12], m[14], m[15]
    ])
    detM02 = m3_determinant([
        m[4], m[5], m[7],
        m[8], m[9], m[11],
        m[12], m[13], m[15]
    ])
    detM03 = m3_determinant([
        m[4], m[5], m[6],
        m[8], m[9], m[10],
        m[12], m[13], m[14]
    ])
    #
    detM10 = m3_determinant([
        m[1], m[2], m[3],
        m[9], m[10], m[11],
        m[13], m[14], m[15]
    ])
    detM11 = m3_determinant([
        m[0], m[2], m[3],
        m[8], m[10], m[11],
        m[12], m[14], m[15]
    ])
    detM12 = m3_determinant([
        m[0], m[1], m[3],
        m[8], m[9], m[11],
        m[12], m[13], m[15]
    ])
    detM13 = m3_determinant([
        m[0], m[1], m[2],
        m[8], m[9], m[10],
        m[12], m[13], m[14]
    ])
    #
    detM20 = m3_determinant([
        m[1], m[2], m[3],
        m[5], m[6], m[7],
        m[13], m[14], m[15]
    ])
    detM21 = m3_determinant([
        m[0], m[2], m[3],
        m[4], m[6], m[7],
        m[12], m[14], m[15]
    ])
    detM22 = m3_determinant([
        m[0], m[1], m[3],
        m[4], m[5], m[7],
        m[12], m[13], m[15]
    ])
    detM23 = m3_determinant([
        m[0], m[1], m[2],
        m[4], m[5], m[6],
        m[12], m[13], m[14]
    ])
    #
    detM30 = m3_determinant([
        m[1], m[2], m[3],
        m[5], m[6], m[7],
        m[9], m[10], m[11]
    ])
    detM31 = m3_determinant([
        m[0], m[2], m[3],
        m[4], m[6], m[7],
        m[8], m[10], m[11]
    ])
    detM32 = m3_determinant([
        m[0], m[1], m[3],
        m[4], m[5], m[7],
        m[8], m[9], m[11]
    ])
    detM33 = m3_determinant([
        m[0], m[1], m[2],
        m[4], m[5], m[6],
        m[8], m[9], m[10]
    ])

    # Cofactor matrix
    mCo = [
        detM00, -detM01, detM02, -detM03,
        -detM10, detM11, -detM12, detM13,
        detM20, -detM21, detM22, -detM23,
        -detM30, detM31, -detM32, detM33,

    ]
    # Adjoint matrix
    mAdj = [
        detM00, -detM10, detM20, -detM30,
        -detM01, detM11, -detM21, detM31,
        detM02, -detM12, detM22, -detM32,
        -detM03, detM13, -detM23, detM33,

    ]

    # divide by determinant of original matrix. We mulitply by the inverse
    # so we only divide once:
    detInv = 1 / m4_determinant(m)
    return [
        mAdj[0] * detInv, mAdj[1] * detInv, mAdj[2] * detInv, mAdj[3] * detInv,
        mAdj[4] * detInv, mAdj[5] * detInv, mAdj[6] * detInv, mAdj[7] * detInv,
        mAdj[8] * detInv, mAdj[9] * detInv, mAdj[10] * detInv, mAdj[11] * detInv,
        mAdj[12] * detInv, mAdj[13] * detInv, mAdj[14] * detInv, mAdj[15] * detInv,

    ]
