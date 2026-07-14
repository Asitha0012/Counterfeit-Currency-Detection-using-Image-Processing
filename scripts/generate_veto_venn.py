import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted

def generate_veto_venn():
    plt.figure(figsize=(8, 8))
    
    # Subsets: (a, b, ab, c, ac, bc, abc)
    # F1 (a), F7 (b), F11 (c)
    # Values: a=10, b=10, ab=30, c=20, ac=90, bc=230, abc=1110
    subsets = (10, 10, 30, 20, 90, 230, 1110)
    
    v = venn3_unweighted(subsets=subsets, set_labels=('F1:\nMicro-printing', 'F7:\nWatermark', 'F11:\nSecurity Thread'))
    
    # Set colors
    if v.get_patch_by_id('100'): v.get_patch_by_id('100').set_color('#ff9999')
    if v.get_patch_by_id('010'): v.get_patch_by_id('010').set_color('#66b3ff')
    if v.get_patch_by_id('001'): v.get_patch_by_id('001').set_color('#99ff99')
    if v.get_patch_by_id('110'): v.get_patch_by_id('110').set_color('#cc99ff')
    if v.get_patch_by_id('101'): v.get_patch_by_id('101').set_color('#ffff99')
    if v.get_patch_by_id('011'): v.get_patch_by_id('011').set_color('#ffb366')
    if v.get_patch_by_id('111'): v.get_patch_by_id('111').set_color('#ff6666')
    
    # Add outlines
    # venn3_circles is not compatible with unweighted in this version, relying on patch borders instead.
    plt.title("Veto Gate System Effectiveness: Venn Diagram\n(Intersection of Fake Notes Caught, Total = 1500)", fontsize=14, fontweight='bold', pad=20)
    
    # Add a text box explaining the total
    textstr = 'Total Fakes Tested: 1500\nTotal Escaped: 0\nIntercept Rate: 100%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', bbox=props)
             
    plt.tight_layout()
    plt.savefig('Veto_Gate_Venn.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating Veto Gate Venn Diagram...")
    generate_veto_venn()
    print("Venn Diagram generated successfully.")
