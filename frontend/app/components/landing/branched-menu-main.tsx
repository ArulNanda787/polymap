"use client"
import BranchedMenu from '@/components/BranchedMenu';
import { useRouter } from 'next/navigation';

export default function NavMenu() {
    const router = useRouter();

    return (
        <BranchedMenu
            items={[
                {
                    label: 'Polymap',
                    children: [
                        {
                            value: 'capstone',
                            label: 'Capstone'
                        }
                    ]
                },
                {
                    label: 'Done by:',
                    children: [
                        { value: 'rudro', label: 'Rudro' },
                        { value: 'arul', label: 'Arul' },
                        { value: 'manav', label: 'Manav' }
                    ]
                }
            ]}
            defaultOpen={[0, 1]}
            defaultActive="capstone"
            onSelect={(value) => {
                if (value === 'capstone') {
                    router.push('/dashboard');
                }
            }}
            color="#000000"
            accentColor="#000000"
            lineColor="#d4d4d8"
            width={260}              // Slightly wider to accommodate larger text
            rowHeight={42}           // Increased from 36 to give taller rows for bigger text
            indent={44}              // Pushed the child text slightly further right
            trunk={14}
            radius={10}
            lineWidth={1.5}
            fontSize={18}            // Increased from 14 to 18
            drawDuration={400}
            foldDuration={300}
        />
    );
}