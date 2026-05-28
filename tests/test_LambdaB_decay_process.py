import pytest
import numpy as np
import hadron2np


class TestLambdaBDecayProcessWidth:
    """测试 Lambda_b 重子衰变过程及其宽度计算"""

    @pytest.fixture
    def parameters(self):
        """获取参数字典"""
        return hadron2np.parameters_dict

    # ==================== 标量暗物质 phi ====================
    
    @pytest.mark.skip(reason="Lambda_b -> Lambda + phi 单粒子过程尚未实现")
    def test_Lambdab_to_Lambda_single_phi(self, parameters):
        """测试 Lambda_b -> Lambda + phi (单标量暗物质)"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'phi'],
            basis='DLEFT(S/P)',
            ff_imp='LQCD-2016-nominal'
        )
        
        # 验证过程对象创建成功
        assert process is not None
        assert process.IS == 'Lambda_b'
        assert process.FS == 'Lambda'
        assert process.dms == ['phi']
        assert isinstance(process, hadron2np.TwoBodyDecayProcess)
        
        # 两体衰变不需要Wilson系数
        wcs = {}
        
        # 测试五组不同的暗物质质量
        m_dm_values = [
            [0],              # 质量为零
            [0.1],            # 质量非零
            [0.5],            # 质量较大
            [1.0],            # 质量更大
            [2.0]             # 质量很大
        ]
        
        for m_dm in m_dm_values:
            width = process.width(wcs=wcs, m_dm=m_dm)
            # 只判断宽度是非负实数
            assert isinstance(width, (float, np.floating))
            assert width >= 0
            
            # 计算分支比
            br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
            # 只判断分支比是非负实数
            assert isinstance(br, (float, np.floating))
            assert br >= 0

    def test_Lambdab_to_Lambda_double_phi(self, parameters):
        """测试 Lambda_b -> Lambda + phi + phi (双标量暗物质)"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'phi', 'phi'],
            basis='DLEFT(S/P)',
            ff_imp='LQCD-2016-nominal'
        )
        
        # 验证过程对象创建成功
        assert process is not None
        assert process.IS == 'Lambda_b'
        assert process.FS == 'Lambda'
        assert process.dms == ['phi', 'phi']
        assert isinstance(process, hadron2np.ThreeBodyDecayProcess)
        
        # 设置 Wilson 系数
        wc_here = np.zeros((3, 3, 2, 2), dtype=complex)
        wc_here[*process.index] = 0.5
        wcs = {'L_S_dphi2': wc_here}
        
        # 测试五组不同的暗物质质量
        m_dm_values = [
            [0, 0],           # 质量相同且为零
            [0.1, 0.1],       # 质量相同且非零
            [0.5, 0.5],       # 质量相同且较大
            [0.1, 0.5],       # 质量不同
            [0.2, 0.8]        # 质量不同且差异较大
        ]
        
        for m_dm in m_dm_values:
            width = process.width(wcs=wcs, m_dm=m_dm)
            # 只判断宽度是非负实数
            assert isinstance(width, (float, np.floating))
            assert width >= 0
            
            # 计算分支比
            br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
            # 只判断分支比是非负实数
            assert isinstance(br, (float, np.floating))
            assert br >= 0

    # ==================== 矢量暗物质 X ====================
    
    @pytest.mark.skip(reason="Lambda_b -> Lambda + X 单粒子过程尚未实现且矢量暗物质X的Wilson系数定义不完整")
    def test_Lambdab_to_Lambda_single_X(self, parameters):
        """测试 Lambda_b -> Lambda + X (单矢量暗物质)"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'X'],
            basis='DLEFT(S/P)',
            ff_imp='LQCD-2016-nominal'
        )
        
        # 验证过程对象创建成功
        assert process is not None
        assert process.IS == 'Lambda_b'
        assert process.FS == 'Lambda'
        assert process.dms == ['X']
        assert isinstance(process, hadron2np.TwoBodyDecayProcess)
        
        # 两体衰变不需要Wilson系数
        wcs = {}
        
        # 测试五组不同的暗物质质量
        m_dm_values = [
            [0],              # 质量为零
            [0.1],            # 质量非零
            [0.5],            # 质量较大
            [1.0],            # 质量更大
            [2.0]             # 质量很大
        ]
        
        for m_dm in m_dm_values:
            width = process.width(wcs=wcs, m_dm=m_dm)
            # 只判断宽度是非负实数
            assert isinstance(width, (float, np.floating))
            assert width >= 0
            
            # 计算分支比
            br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
            # 只判断分支比是非负实数
            assert isinstance(br, (float, np.floating))
            assert br >= 0

    @pytest.mark.skip(reason="矢量暗物质X的Wilson系数定义不完整,缺少L_Ppartial_dX2")
    def test_Lambdab_to_Lambda_double_X(self, parameters):
        """测试 Lambda_b -> Lambda + X + X (双矢量暗物质)"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'X', 'X'],
            basis='DLEFT(S/P)',
            ff_imp='LQCD-2016-nominal'
        )
        
        # 验证过程对象创建成功
        assert process is not None
        assert process.IS == 'Lambda_b'
        assert process.FS == 'Lambda'
        assert process.dms == ['X', 'X']
        assert isinstance(process, hadron2np.ThreeBodyDecayProcess)
        
        # 设置 Wilson 系数
        wc_here = np.zeros((3, 3, 2, 2), dtype=complex)
        wc_here[*process.index] = 0.5
        wcs = {'L_S_dX2': wc_here}
        
        # 测试五组不同的暗物质质量
        m_dm_values = [
            [0, 0],           # 质量相同且为零
            [0.1, 0.1],       # 质量相同且非零
            [0.5, 0.5],       # 质量相同且较大
            [0.1, 0.5],       # 质量不同
            [0.2, 0.8]        # 质量不同且差异较大
        ]
        
        for m_dm in m_dm_values:
            width = process.width(wcs=wcs, m_dm=m_dm)
            # 只判断宽度是非负实数
            assert isinstance(width, (float, np.floating))
            assert width >= 0
            
            # 计算分支比
            br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
            # 只判断分支比是非负实数
            assert isinstance(br, (float, np.floating))
            assert br >= 0

    # ==================== 费米子暗物质 chi (mode 1: chi-chi) ====================
    
    def test_Lambdab_to_Lambda_fermion_chi_chi_mode1(self, parameters):
        """测试 Lambda_b -> Lambda + chi + chi (费米子暗物质 mode 1)"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'chi', 'chi'],
            basis='DLEFT(S/P)',
            ff_imp='LQCD-2016-nominal'
        )
        
        # 验证过程对象创建成功
        assert process is not None
        assert process.IS == 'Lambda_b'
        assert process.FS == 'Lambda'
        assert process.dms == ['chi', 'chi']
        assert process.dm_mode == 1  # mode 1: chi-chi
        assert isinstance(process, hadron2np.ThreeBodyDecayProcess)
        
        # 设置 Wilson 系数
        wc_here = np.zeros((3, 3, 2, 2), dtype=complex)
        wc_here[*process.index] = 0.5
        wcs = {'L_S_dchi2': wc_here}
        
        # 测试五组不同的暗物质质量
        m_dm_values = [
            [0, 0],           # 质量相同且为零
            [0.1, 0.1],       # 质量相同且非零
            [0.5, 0.5],       # 质量相同且较大
            [0.1, 0.5],       # 质量不同
            [0.2, 0.8]        # 质量不同且差异较大
        ]
        
        for m_dm in m_dm_values:
            width = process.width(wcs=wcs, m_dm=m_dm)
            # 只判断宽度是非负实数
            assert isinstance(width, (float, np.floating))
            assert width >= 0
            
            # 计算分支比
            br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
            # 只判断分支比是非负实数
            assert isinstance(br, (float, np.floating))
            assert br >= 0

    # ==================== 费米子暗物质 (mode 2: chibar-chi) ====================
    
    def test_Lambdab_to_Lambda_fermion_chibar_chi_mode2(self, parameters):
        """测试 Lambda_b -> Lambda + chibar + chi (费米子暗物质 mode 2)"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'chibar', 'chi'],
            basis='DLEFT(S/P)',
            ff_imp='LQCD-2016-nominal'
        )
        
        # 验证过程对象创建成功
        assert process is not None
        assert process.IS == 'Lambda_b'
        assert process.FS == 'Lambda'
        assert process.dms == ['chibar', 'chi']
        assert process.dm_mode == 2  # mode 2: chibar-chi
        assert isinstance(process, hadron2np.ThreeBodyDecayProcess)
        
        # 设置 Wilson 系数
        wc_here = np.zeros((3, 3, 2, 2), dtype=complex)
        wc_here[*process.index] = 0.5
        wcs = {'L_S_dchi2': wc_here}
        
        # 测试五组不同的暗物质质量
        m_dm_values = [
            [0, 0],           # 质量相同且为零
            [0.1, 0.1],       # 质量相同且非零
            [0.5, 0.5],       # 质量相同且较大
            [0.1, 0.5],       # 质量不同
            [0.2, 0.8]        # 质量不同且差异较大
        ]
        
        for m_dm in m_dm_values:
            width = process.width(wcs=wcs, m_dm=m_dm)
            # 只判断宽度是非负实数
            assert isinstance(width, (float, np.floating))
            assert width >= 0
            
            # 计算分支比
            br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
            # 只判断分支比是非负实数
            assert isinstance(br, (float, np.floating))
            assert br >= 0

    # ==================== 费米子暗物质 (mode 3: chibar-chibar) ====================
    
    def test_Lambdab_to_Lambda_fermion_chibar_chibar_mode3(self, parameters):
        """测试 Lambda_b -> Lambda + chibar + chibar (费米子暗物质 mode 3)"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'chibar', 'chibar'],
            basis='DLEFT(S/P)',
            ff_imp='LQCD-2016-nominal'
        )
        
        # 验证过程对象创建成功
        assert process is not None
        assert process.IS == 'Lambda_b'
        assert process.FS == 'Lambda'
        assert process.dms == ['chibar', 'chibar']
        assert process.dm_mode == 3  # mode 3: chibar-chibar
        assert isinstance(process, hadron2np.ThreeBodyDecayProcess)
        
        # 设置 Wilson 系数
        wc_here = np.zeros((3, 3, 2, 2), dtype=complex)
        wc_here[*process.index] = 0.5
        wcs = {'L_S_dchi2': wc_here}
        
        # 测试五组不同的暗物质质量
        m_dm_values = [
            [0, 0],           # 质量相同且为零
            [0.1, 0.1],       # 质量相同且非零
            [0.5, 0.5],       # 质量相同且较大
            [0.1, 0.5],       # 质量不同
            [0.2, 0.8]        # 质量不同且差异较大
        ]
        
        for m_dm in m_dm_values:
            width = process.width(wcs=wcs, m_dm=m_dm)
            # 只判断宽度是非负实数
            assert isinstance(width, (float, np.floating))
            assert width >= 0
            
            # 计算分支比
            br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
            # 只判断分支比是非负实数
            assert isinstance(br, (float, np.floating))
            assert br >= 0

    # ==================== 标准模型过程中微子 nu ====================
    
    def test_Lambdab_to_Lambda_nu_nu_SM_process(self, parameters):
        """特别测试 Lambda_b -> Lambda nu nu 标准模型过程"""
        process = hadron2np.new_decay_process(
            ['Lambda_b', 'Lambda', 'nu', 'nu'], 
            basis='DLEFT(S/P)', 
            ff_imp='LQCD-2016-nominal'
        )
        
        assert process.dm_name == 'nu'
        assert process.dms == ['nu', 'nu']
        
        # 标准模型过程使用空 Wilson 系数
        wcs = {}
        m_dm = [0, 0]
        
        width = process.width(wcs=wcs, m_dm=m_dm)
        assert isinstance(width, (float, np.floating))
        assert width >= 0
        
        # 计算分支比
        br = process.branching_ratio(wcs=wcs, m_dm=m_dm)
        assert isinstance(br, (float, np.floating))
        assert br >= 0
        
        # 验证与 Lambda_b 总宽度的比值
        br_normalized = width / parameters['Gamma_Lambda_b']
        assert isinstance(br_normalized, (float, np.floating))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
