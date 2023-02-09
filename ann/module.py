from ann_predict_module import func
people, normal, hand, not_normal_hand, not_normal= func()
print("總人數:",people,"  ")
print("站姿人數:",normal)
print("站姿舉手人數:",hand)
print("坐姿舉手人數:",not_normal_hand)
print("坐姿人數:",not_normal)
