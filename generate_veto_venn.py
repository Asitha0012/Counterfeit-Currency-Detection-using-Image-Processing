import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted, venn3_circles

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def generate_veto_venn(dark_mode=False):
    plt.figure(figsize=(8, 8))
    
    subsets = (10, 10, 30, 20, 90, 230, 1110)
    
    v = venn3_unweighted(subsets=subsets, set_labels=('F1:\nMicro-printing', 'F7:\nWatermark', 'F11:\nSecurity Thread'))
    
    if dark_mode:
        colors = {'100': '#990000', '010': '#003399', '001': '#006600', 
                  '110': '#4c0099', '101': '#997300', '011': '#994c00', '111': '#4d0000'}
    else:
        colors = {'100': '#ff9999', '010': '#66b3ff', '001': '#99ff99', 
                  '110': '#cc99ff', '101': '#ffff99', '011': '#ffb366', '111': '#ff6666'}

    for subset_id, color in colors.items():
        patch = v.get_patch_by_id(subset_id)
        if patch: 
            patch.set_color(color)
            if dark_mode:
                patch.set_alpha(0.85)

    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='solid', linewidth=1.5, color='#333333')
    
    extra_artists = []
    
    text_size = 20
    number_size = 20

    for t in v.set_labels:
        if t:
            t.set_fontsize(text_size)
            extra_artists.append(t)
            
    for t in v.subset_labels:
        if t:
            t.set_fontsize(number_size)
            if dark_mode:
                t.set_color('white')
            extra_artists.append(t)

    textstr = 'Total Fakes Tested: 1500\nTotal Escaped: 0\nIntercept Rate: 100%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    
    text_box = plt.text(-0.20, -0.02, textstr, transform=plt.gca().transAxes, fontsize=text_size,
             verticalalignment='bottom', horizontalalignment='left', bbox=props)
    extra_artists.append(text_box)
             
    plt.tight_layout()
    suffix = '_Dark' if dark_mode else '_Light'
    plt.savefig(f'Veto_Gate_Venn{suffix}.pdf', dpi=300, bbox_extra_artists=extra_artists, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == '__main__':
    print('Generating Veto Gate Venn Diagrams...')
    generate_veto_venn(dark_mode=False)
    generate_veto_venn(dark_mode=True)
    print('Venn Diagrams generated successfully.')
